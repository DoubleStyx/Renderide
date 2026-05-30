//! Instance grouping for world-mesh forward draws.
//!
//! Produces an [`InstancePlan`] that groups `(batch_key, mesh, submesh)` runs into a
//! contiguous per-draw-slab range regardless of where the sort placed individual members.
//! The forward pass packs the per-draw slab in `slab_layout` order and emits one
//! `draw_indexed(.., 0, instance_range)` per [`DrawGroup`].
//!
//! Replaces the older `(regular_indices, intersect_indices) + for_each_instance_batch`
//! pipeline whose merge requirement was *adjacency in the sorted draw array* -- that policy
//! silently fragmented instancing whenever the sort cascade interleaved same-mesh draws
//! with different-mesh draws (e.g. varying `sorting_order` within one material).

mod batch_window;
mod scratch;

use std::ops::Range;

use hashbrown::HashMap;
use rayon::prelude::*;

use crate::cpu_parallelism::{
    ParallelAdmission, RENDER_COMMAND_CHUNK_DRAWS, admit_render_command_items,
    current_reference_worker_count, record_parallel_admission, reference_worker_count,
};
use crate::materials::{
    RasterPipelineKind, ShaderPermutation, UNITY_RENDER_QUEUE_ALPHA_TEST,
    embedded_stem_depth_prepass_pass,
};
use crate::render_phase::{RenderPhaseKey, RenderPhaseSet};

use super::draw_prep::WorldMeshDrawItem;

use batch_window::{BatchWindow, build_group, draw_requires_singleton, next_batch_window};
use scratch::{InstancePlanScratch, MeshSubmeshKey, mesh_submesh_key};

/// Minimum independent batch windows needed to amortize Rayon scheduling and merge overhead.
const INSTANCE_PLAN_PARALLEL_MIN_WINDOWS: usize = 2;
/// Draw count above which [`build_plan`] may split batch windows across worker threads.
const INSTANCE_PLAN_PARALLEL_MIN_DRAWS: usize = RENDER_COMMAND_CHUNK_DRAWS * 2;
/// Maximum batch windows processed by one parallel worker task.
const INSTANCE_PLAN_PARALLEL_MAX_WINDOWS_PER_TASK: usize = 12;
/// Adaptive window chunks assigned to one Rayon worker leaf.
const INSTANCE_PLAN_PARALLEL_CHUNKS_PER_TASK: usize = 1;
/// Draws from one large same-batch-key window assigned to one worker chunk.
const INSTANCE_PLAN_PARALLEL_WINDOW_DRAW_CHUNK: usize = RENDER_COMMAND_CHUNK_DRAWS;
/// Draw count required before a single large batch window can fan out internally.
const INSTANCE_PLAN_PARALLEL_MIN_SINGLE_WINDOW_DRAWS: usize =
    INSTANCE_PLAN_PARALLEL_WINDOW_DRAW_CHUNK * 2;

/// Worker-local members for one mesh/submesh group inside a large batch window.
struct LocalGroupedWindowGroup {
    /// Mesh/submesh key shared by every member.
    key: MeshSubmeshKey,
    /// First sorted draw index for this group.
    representative_draw_idx: usize,
    /// Sorted draw indexes belonging to the group.
    members: Vec<usize>,
}

/// Worker-local grouping result for one chunk of a large batch window.
struct LocalGroupedWindowChunk {
    /// Groups in first-seen order inside the chunk.
    groups: Vec<LocalGroupedWindowGroup>,
}

/// Merged members for one mesh/submesh group across a whole large batch window.
struct MergedGroupedWindowGroup {
    /// First sorted draw index for this group.
    representative_draw_idx: usize,
    /// Sorted draw indexes belonging to the group.
    members: Vec<usize>,
}

/// Compatibility key for grouping draws after material packet resolution.
#[derive(Clone, Copy, Hash, Eq, PartialEq)]
struct SubmissionGroupKey {
    /// Primary render phase for the group.
    phase: WorldMeshPhase,
    /// Conservative segment split by strict-order barriers inside a phase.
    order_segment: u32,
    /// Resolved material submission class assigned by forward frame preparation.
    submission_class: u32,
    /// Mesh/submesh identity submitted by the indexed draw call.
    mesh: MeshSubmeshKey,
}

/// Pending group built from resolved submission compatibility rather than raw material ids.
struct PendingSubmissionGroup {
    /// Primary render phase for the group.
    phase: WorldMeshPhase,
    /// First sorted draw index in the group.
    representative_draw_idx: usize,
    /// Sorted draw indexes belonging to the group.
    members: Vec<usize>,
}

/// One emitted indexed draw covering a contiguous slab range of identical instances.
///
/// All members of a group share `batch_key`, `mesh_asset_id`, `first_index`, and
/// `index_count` by construction (see [`build_plan`]), so the forward pass can
/// drive material binds, vertex streams, and stencil reference from any single member.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DrawGroup {
    /// Index in the sorted `draws` array of the group's first member in sort order.
    ///
    /// Used by the forward pass to read material/state fields that are uniform across the group.
    pub representative_draw_idx: usize,
    /// Slab-coordinate range to pass as `first_instance..first_instance + count` to
    /// `draw_indexed`. Indexes into [`InstancePlan::slab_layout`], not into `draws`.
    pub instance_range: Range<u32>,
    /// Index into the view's pre-resolved material packet table.
    ///
    /// Filled after material packets are resolved during backend world-mesh frame planning.
    /// Defaults to zero while the grouping plan is being built, then recording consumes it
    /// directly instead of cursoring through packet boundaries per group.
    pub material_packet_idx: usize,
}

/// Named mesh render phase produced by world-mesh instance planning.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum WorldMeshPhase {
    /// Conservative depth-only mirror of eligible pre-skybox forward groups.
    DepthOnly,
    /// Pre-skybox opaque forward groups below the alpha-test queue.
    ForwardOpaque,
    /// Pre-skybox alpha-test forward groups.
    ForwardAlphaTest,
    /// Normal-prepass mirror of pre-skybox forward groups.
    ViewNormals,
    /// Nontransparent intersection-material groups that run after the scene-depth snapshot.
    Intersection,
    /// Post-skybox transparent groups that do not require a grab snapshot.
    Transparent,
    /// Transparent scene-color snapshot filter groups.
    TransparentGrab,
}

impl WorldMeshPhase {
    /// All world-mesh phase keys in dense index order.
    pub const ALL: [Self; 7] = [
        Self::DepthOnly,
        Self::ForwardOpaque,
        Self::ForwardAlphaTest,
        Self::ViewNormals,
        Self::Intersection,
        Self::Transparent,
        Self::TransparentGrab,
    ];

    /// Primary phases that submit visible world-mesh material passes.
    pub const PRIMARY_FORWARD: [Self; 5] = [
        Self::ForwardOpaque,
        Self::ForwardAlphaTest,
        Self::Intersection,
        Self::Transparent,
        Self::TransparentGrab,
    ];
}

impl RenderPhaseKey for WorldMeshPhase {
    const COUNT: usize = Self::ALL.len();

    fn index(self) -> usize {
        match self {
            Self::DepthOnly => 0,
            Self::ForwardOpaque => 1,
            Self::ForwardAlphaTest => 2,
            Self::ViewNormals => 3,
            Self::Intersection => 4,
            Self::Transparent => 5,
            Self::TransparentGrab => 6,
        }
    }
}

/// Mesh pass consumer for one or more world-mesh phases.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MeshPassKind {
    /// Generic safe depth-only prepass.
    DepthPrepass,
    /// Main pre-skybox forward material pass.
    ForwardOpaque,
    /// GTAO view-normal prepass.
    ViewNormals,
    /// Intersection material pass.
    Intersection,
    /// Post-skybox transparent and grab-pass sequence.
    TransparentSequence,
}

impl MeshPassKind {
    /// Returns the phases consumed by this mesh pass in submission order.
    pub fn phases(self) -> &'static [WorldMeshPhase] {
        match self {
            Self::DepthPrepass => &[WorldMeshPhase::DepthOnly],
            Self::ForwardOpaque => &[
                WorldMeshPhase::ForwardOpaque,
                WorldMeshPhase::ForwardAlphaTest,
            ],
            Self::ViewNormals => &[WorldMeshPhase::ViewNormals],
            Self::Intersection => &[WorldMeshPhase::Intersection],
            Self::TransparentSequence => {
                &[WorldMeshPhase::Transparent, WorldMeshPhase::TransparentGrab]
            }
        }
    }

    /// Returns the first phase consumed by this mesh pass.
    pub fn first_phase(self) -> WorldMeshPhase {
        match self {
            Self::DepthPrepass => WorldMeshPhase::DepthOnly,
            Self::ForwardOpaque => WorldMeshPhase::ForwardOpaque,
            Self::ViewNormals => WorldMeshPhase::ViewNormals,
            Self::Intersection => WorldMeshPhase::Intersection,
            Self::TransparentSequence => WorldMeshPhase::Transparent,
        }
    }
}

/// Per-view instance plan: slab layout plus named mesh render phases.
///
/// The forward pass packs the per-draw slab in `slab_layout` order -- slot `i` holds the
/// per-draw uniforms for `draws[slab_layout[i]]` -- and emits each group's `instance_range`
/// directly. `representative_draw_idx` for each group list is monotonically increasing; backend
/// frame planning attaches material packet indices after packet resolution so recording does not
/// search packet boundaries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InstancePlan {
    /// New slab order. `slab_layout[i]` is the sorted-draw index whose per-draw uniforms
    /// go into per-draw slot `i`. Length equals `draws.len()` (every draw gets one slot).
    pub slab_layout: Vec<usize>,
    /// Phase queues keyed by [`WorldMeshPhase`].
    phases: RenderPhaseSet<WorldMeshPhase, DrawGroup>,
}

impl InstancePlan {
    /// Creates an empty plan with all phase queues initialized.
    pub fn new() -> Self {
        Self {
            slab_layout: Vec::new(),
            phases: RenderPhaseSet::new(),
        }
    }

    /// Creates an empty plan sized for `draw_count` slab entries.
    fn with_capacity(draw_count: usize) -> Self {
        Self {
            slab_layout: Vec::with_capacity(draw_count),
            phases: RenderPhaseSet::new(),
        }
    }

    /// Returns the groups queued in `phase`.
    pub fn phase(&self, phase: WorldMeshPhase) -> &[DrawGroup] {
        self.phases.phase(phase).items()
    }

    /// Returns the groups queued in `phase` mutably.
    pub fn phase_mut(&mut self, phase: WorldMeshPhase) -> &mut Vec<DrawGroup> {
        self.phases.phase_mut(phase).items_mut()
    }

    /// Returns whether `phase` has no queued groups.
    pub fn phase_is_empty(&self, phase: WorldMeshPhase) -> bool {
        self.phases.phase(phase).is_empty()
    }

    /// Returns the number of groups queued in `phase`.
    pub fn phase_len(&self, phase: WorldMeshPhase) -> usize {
        self.phases.phase(phase).len()
    }

    /// Returns all primary visible forward groups.
    pub fn primary_forward_groups(&self) -> impl Iterator<Item = &DrawGroup> {
        WorldMeshPhase::PRIMARY_FORWARD
            .iter()
            .flat_map(|&phase| self.phase(phase).iter())
    }

    /// Returns the total number of primary visible forward groups.
    pub fn primary_forward_group_count(&self) -> usize {
        WorldMeshPhase::PRIMARY_FORWARD
            .iter()
            .map(|&phase| self.phase_len(phase))
            .sum()
    }
}

impl Default for InstancePlan {
    fn default() -> Self {
        Self::new()
    }
}

/// Builds the per-view [`InstancePlan`] from a sorted draw list.
///
/// Same-`batch_key` runs are already adjacent because of the sort, so grouping happens in a
/// small per-window `HashMap<MeshSubmeshKey, group_idx>` that is cleared between windows. Large
/// plans with enough independent material windows build those windows on Rayon workers and merge
/// the resulting slab ranges in sorted-window order. Singleton-per-draw groups are produced when:
/// - `supports_base_instance` is false (downlevel devices set `instance_count == 1`), or
/// - the run is `skinned` (vertex deform path differs per draw), or
/// - the run is order-dependent transparency whose back-to-front order is load-bearing.
///
/// Group emit order matches the order of each group's first member in `draws`, so the
/// view's high-level sort intent (state-change minimisation, transparent depth) is
/// preserved while same-mesh members that landed later still merge in.
#[cfg(test)]
pub fn build_plan(draws: &[WorldMeshDrawItem], supports_base_instance: bool) -> InstancePlan {
    build_plan_for_shader(draws, supports_base_instance, ShaderPermutation(0))
}

/// Builds a per-view [`InstancePlan`] using the active shader permutation.
pub fn build_plan_for_shader(
    draws: &[WorldMeshDrawItem],
    supports_base_instance: bool,
    shader_perm: ShaderPermutation,
) -> InstancePlan {
    profiling::scope!("mesh::build_plan");
    if draws.is_empty() {
        return InstancePlan::default();
    }

    if draws.len() < INSTANCE_PLAN_PARALLEL_MIN_DRAWS {
        record_parallel_admission(
            "world_mesh_instance_plan",
            draws.len(),
            draws.len(),
            ParallelAdmission::Serial,
        );
        return build_plan_serial(draws, supports_base_instance, shader_perm);
    }

    let windows = collect_batch_windows(draws, supports_base_instance);
    let worker_count = current_reference_worker_count();
    let admission = if windows.len() == 1 {
        single_window_admission(windows[0].range.len(), worker_count)
    } else {
        instance_plan_admission(draws.len(), windows.len(), worker_count)
    };
    record_parallel_admission(
        "world_mesh_instance_plan",
        draws.len(),
        if windows.len() == 1 {
            draws.len()
        } else {
            windows.len()
        },
        admission,
    );
    if windows.len() == 1 && admission.is_parallel() {
        return build_plan_from_large_window_parallel(draws, &windows[0], shader_perm);
    }
    if admission.is_parallel() {
        build_plan_parallel(draws, &windows, shader_perm)
    } else {
        build_plan_from_windows_serial(draws, &windows, shader_perm)
    }
}

/// Builds a per-view [`InstancePlan`] from pre-resolved material submission compatibility.
///
/// `submission_classes[i]` must identify the concrete pipeline and group-1 material binding that
/// will be submitted for `draws[i]`. This lets equivalent materials share GPU instance batches even
/// when their source material or property-block ids differ. Strict transparent/grab ordering,
/// skinned draws, and devices without base-instance support still emit singleton groups.
pub fn build_plan_for_shader_with_submission_classes(
    draws: &[WorldMeshDrawItem],
    submission_classes: &[u32],
    supports_base_instance: bool,
    shader_perm: ShaderPermutation,
) -> InstancePlan {
    profiling::scope!("mesh::build_plan_submission_classes");
    debug_assert_eq!(
        draws.len(),
        submission_classes.len(),
        "draw submission classes should align with sorted draws",
    );
    if draws.is_empty() {
        return InstancePlan::default();
    }
    if draws.len() != submission_classes.len() {
        return build_plan_for_shader(draws, supports_base_instance, shader_perm);
    }

    build_plan_from_submission_classes_serial(
        draws,
        submission_classes,
        supports_base_instance,
        shader_perm,
    )
}

/// Builds a submission-class plan with deterministic first-seen group order.
fn build_plan_from_submission_classes_serial(
    draws: &[WorldMeshDrawItem],
    submission_classes: &[u32],
    supports_base_instance: bool,
    shader_perm: ShaderPermutation,
) -> InstancePlan {
    let mut group_index: HashMap<SubmissionGroupKey, usize> = HashMap::new();
    let mut pending_groups: Vec<PendingSubmissionGroup> = Vec::new();
    let mut order_segments = [0u32; WorldMeshPhase::ALL.len()];

    for (draw_idx, item) in draws.iter().enumerate() {
        let classification =
            crate::world_mesh::phase_classification::classify_world_mesh_batch(&item.batch_key);
        let phase = classification.phase;
        if draw_requires_singleton(item, supports_base_instance) {
            pending_groups.push(PendingSubmissionGroup {
                phase,
                representative_draw_idx: draw_idx,
                members: vec![draw_idx],
            });
            if classification.strict_order || classification.grab_pass {
                let segment = &mut order_segments[phase.index()];
                *segment = segment.saturating_add(1);
            }
            continue;
        }

        let key = SubmissionGroupKey {
            phase,
            order_segment: order_segments[phase.index()],
            submission_class: submission_classes[draw_idx],
            mesh: mesh_submesh_key(item),
        };
        if let Some(&group_idx) = group_index.get(&key) {
            pending_groups[group_idx].members.push(draw_idx);
        } else {
            let group_idx = pending_groups.len();
            group_index.insert(key, group_idx);
            pending_groups.push(PendingSubmissionGroup {
                phase,
                representative_draw_idx: draw_idx,
                members: vec![draw_idx],
            });
        }
    }

    let mut builder = InstancePlanBuilder::with_capacity(draws.len(), shader_perm);
    builder.emit_submission_groups(draws, pending_groups);
    builder.finish()
}

/// Returns whether instance planning should use its active Rayon pool.
#[cfg(test)]
fn should_parallelize_instance_plan(draw_count: usize, window_count: usize) -> bool {
    should_parallelize_instance_plan_with_workers(
        draw_count,
        window_count,
        current_reference_worker_count(),
    )
}

/// Returns the admission decision for one large same-batch-key window.
fn single_window_admission(draw_count: usize, worker_count: usize) -> ParallelAdmission {
    let admission = admit_render_command_items(draw_count, worker_count);
    if draw_count >= INSTANCE_PLAN_PARALLEL_MIN_SINGLE_WINDOW_DRAWS && admission.is_parallel() {
        admission
    } else {
        ParallelAdmission::Serial
    }
}

/// Returns whether instance planning has enough work and workers to split batch windows.
#[cfg(test)]
fn should_parallelize_instance_plan_with_workers(
    draw_count: usize,
    window_count: usize,
    worker_count: usize,
) -> bool {
    instance_plan_admission(draw_count, window_count, worker_count).is_parallel()
}

/// Returns the admission decision for splitting batch windows across workers.
fn instance_plan_admission(
    draw_count: usize,
    window_count: usize,
    worker_count: usize,
) -> ParallelAdmission {
    let draw_admission = admit_render_command_items(draw_count, worker_count);
    if window_count >= INSTANCE_PLAN_PARALLEL_MIN_WINDOWS
        && draw_admission.is_parallel()
        && parallel_window_chunk_count_with_workers(window_count, worker_count) >= 2
    {
        ParallelAdmission::Parallel {
            chunk_size: INSTANCE_PLAN_PARALLEL_CHUNKS_PER_TASK,
        }
    } else {
        ParallelAdmission::Serial
    }
}

/// Returns how many worker chunks `window_count` produces for a known worker count.
fn parallel_window_chunk_count_with_workers(window_count: usize, worker_count: usize) -> usize {
    window_count.div_ceil(parallel_window_chunk_size_with_workers(
        window_count,
        worker_count,
    ))
}

/// Returns the batch-window chunk size to use with the active Rayon pool.
fn parallel_window_chunk_size(window_count: usize) -> usize {
    parallel_window_chunk_size_with_workers(window_count, rayon::current_num_threads())
}

/// Returns an adaptive batch-window chunk size capped to avoid over-coalescing work.
fn parallel_window_chunk_size_with_workers(window_count: usize, worker_count: usize) -> usize {
    let worker_count = reference_worker_count(worker_count);
    window_count
        .div_ceil(worker_count)
        .clamp(1, INSTANCE_PLAN_PARALLEL_MAX_WINDOWS_PER_TASK)
}

fn build_plan_serial(
    draws: &[WorldMeshDrawItem],
    supports_base_instance: bool,
    shader_perm: ShaderPermutation,
) -> InstancePlan {
    let mut builder = InstancePlanBuilder::with_capacity(draws.len(), shader_perm);
    let mut i = 0usize;
    while i < draws.len() {
        let window = next_batch_window(draws, i, supports_base_instance);
        i = window.range.end;
        builder.process_window(draws, window);
    }

    builder.finish()
}

fn collect_batch_windows(
    draws: &[WorldMeshDrawItem],
    supports_base_instance: bool,
) -> Vec<BatchWindow> {
    let mut windows = Vec::new();
    let mut i = 0usize;
    while i < draws.len() {
        let window = next_batch_window(draws, i, supports_base_instance);
        i = window.range.end;
        windows.push(window);
    }
    windows
}

fn build_plan_from_windows_serial(
    draws: &[WorldMeshDrawItem],
    windows: &[BatchWindow],
    shader_perm: ShaderPermutation,
) -> InstancePlan {
    let mut builder = InstancePlanBuilder::with_capacity(draws.len(), shader_perm);
    for window in windows {
        builder.process_window(draws, window.clone());
    }
    builder.finish()
}

fn build_plan_parallel(
    draws: &[WorldMeshDrawItem],
    windows: &[BatchWindow],
    shader_perm: ShaderPermutation,
) -> InstancePlan {
    profiling::scope!("mesh::build_plan_parallel_windows");
    let partials: Vec<_> = {
        profiling::scope!("mesh::build_plan_parallel_windows::worker_chunks");
        let window_chunk_size = parallel_window_chunk_size(windows.len());
        windows
            .par_chunks(window_chunk_size)
            .with_min_len(INSTANCE_PLAN_PARALLEL_CHUNKS_PER_TASK)
            .map(|chunk| {
                profiling::scope!("mesh::build_plan_window_chunk_worker");
                let draw_count = chunk.iter().map(|window| window.range.len()).sum::<usize>();
                let mut builder = InstancePlanBuilder::with_capacity(draw_count, shader_perm);
                for window in chunk {
                    builder.process_window(draws, window.clone());
                }
                builder.finish()
            })
            .collect()
    };
    {
        profiling::scope!("mesh::build_plan_parallel_windows::merge");
        merge_partial_instance_plans(draws.len(), partials)
    }
}

/// Builds a plan for one large same-batch-key window by splitting the window itself.
fn build_plan_from_large_window_parallel(
    draws: &[WorldMeshDrawItem],
    window: &BatchWindow,
    shader_perm: ShaderPermutation,
) -> InstancePlan {
    profiling::scope!("mesh::build_plan_parallel_large_window");
    if window.singleton {
        build_singleton_window_parallel(draws, window, shader_perm)
    } else {
        build_grouped_window_parallel(draws, window, shader_perm)
    }
}

/// Builds singleton groups for a large window in ordered draw chunks.
fn build_singleton_window_parallel(
    draws: &[WorldMeshDrawItem],
    window: &BatchWindow,
    shader_perm: ShaderPermutation,
) -> InstancePlan {
    let ranges = window_draw_chunks(window.range.clone());
    let partials: Vec<_> = ranges
        .par_iter()
        .with_min_len(INSTANCE_PLAN_PARALLEL_CHUNKS_PER_TASK)
        .map(|range| {
            profiling::scope!("mesh::build_plan_singleton_window_chunk_worker");
            let mut builder = InstancePlanBuilder::with_capacity(range.len(), shader_perm);
            builder.process_window(
                draws,
                BatchWindow {
                    range: range.clone(),
                    phase: window.phase,
                    singleton: true,
                },
            );
            builder.finish()
        })
        .collect();
    merge_partial_instance_plans(draws.len(), partials)
}

/// Builds grouped draw batches for a large window by reducing worker-local mesh/submesh groups.
fn build_grouped_window_parallel(
    draws: &[WorldMeshDrawItem],
    window: &BatchWindow,
    shader_perm: ShaderPermutation,
) -> InstancePlan {
    let ranges = window_draw_chunks(window.range.clone());
    let chunks = ranges
        .par_iter()
        .with_min_len(INSTANCE_PLAN_PARALLEL_CHUNKS_PER_TASK)
        .map(|range| {
            profiling::scope!("mesh::build_plan_grouped_window_chunk_worker");
            collect_grouped_window_chunk(draws, range.clone())
        })
        .collect::<Vec<_>>();
    let groups = merge_grouped_window_chunks(chunks);
    let mut builder = InstancePlanBuilder::with_capacity(window.range.len(), shader_perm);
    builder.emit_merged_grouped_window(draws, window.phase, groups);
    builder.finish()
}

/// Splits one batch-window draw range into deterministic draw chunks.
fn window_draw_chunks(range: Range<usize>) -> Vec<Range<usize>> {
    let mut ranges = Vec::new();
    let mut start = range.start;
    while start < range.end {
        let end = (start + INSTANCE_PLAN_PARALLEL_WINDOW_DRAW_CHUNK).min(range.end);
        ranges.push(start..end);
        start = end;
    }
    ranges
}

/// Builds worker-local mesh/submesh groups for a draw range.
fn collect_grouped_window_chunk(
    draws: &[WorldMeshDrawItem],
    range: Range<usize>,
) -> LocalGroupedWindowChunk {
    let mut group_index: HashMap<MeshSubmeshKey, usize> = HashMap::new();
    let mut groups: Vec<LocalGroupedWindowGroup> = Vec::new();
    for draw_idx in range {
        let key = mesh_submesh_key(&draws[draw_idx]);
        if let Some(&group_idx) = group_index.get(&key) {
            groups[group_idx].members.push(draw_idx);
        } else {
            let group_idx = groups.len();
            group_index.insert(key, group_idx);
            groups.push(LocalGroupedWindowGroup {
                key,
                representative_draw_idx: draw_idx,
                members: vec![draw_idx],
            });
        }
    }
    LocalGroupedWindowChunk { groups }
}

/// Merges worker-local groups in draw-chunk order, preserving serial first-seen group order.
fn merge_grouped_window_chunks(
    chunks: Vec<LocalGroupedWindowChunk>,
) -> Vec<MergedGroupedWindowGroup> {
    let mut group_index: HashMap<MeshSubmeshKey, usize> = HashMap::new();
    let mut groups: Vec<MergedGroupedWindowGroup> = Vec::new();
    for chunk in chunks {
        for mut local in chunk.groups {
            if let Some(&group_idx) = group_index.get(&local.key) {
                groups[group_idx].members.append(&mut local.members);
            } else {
                let group_idx = groups.len();
                group_index.insert(local.key, group_idx);
                groups.push(MergedGroupedWindowGroup {
                    representative_draw_idx: local.representative_draw_idx,
                    members: local.members,
                });
            }
        }
    }
    groups
}

fn merge_partial_instance_plans(
    draw_count: usize,
    mut partials: Vec<InstancePlan>,
) -> InstancePlan {
    let mut plan = InstancePlan::with_capacity(draw_count);

    for partial in &mut partials {
        let slab_offset = plan.slab_layout.len() as u32;
        for phase in WorldMeshPhase::ALL {
            append_groups_with_slab_offset(
                plan.phase_mut(phase),
                std::mem::take(partial.phase_mut(phase)),
                slab_offset,
            );
        }
        plan.slab_layout.append(&mut partial.slab_layout);
    }

    debug_assert_plan_group_order(&plan);
    plan
}

fn append_groups_with_slab_offset(
    target: &mut Vec<DrawGroup>,
    groups: Vec<DrawGroup>,
    slab_offset: u32,
) {
    target.extend(groups.into_iter().map(|mut group| {
        group.instance_range =
            group.instance_range.start + slab_offset..group.instance_range.end + slab_offset;
        group
    }));
}

fn debug_assert_plan_group_order(plan: &InstancePlan) {
    for phase in WorldMeshPhase::ALL {
        debug_assert!(groups_are_monotonic(plan.phase(phase)));
    }
}

fn groups_are_monotonic(groups: &[DrawGroup]) -> bool {
    groups
        .windows(2)
        .all(|w| w[0].representative_draw_idx <= w[1].representative_draw_idx)
}

/// Returns whether a regular draw group may be mirrored by the generic opaque depth prepass.
pub(crate) fn depth_prepass_group_eligible(
    draws: &[WorldMeshDrawItem],
    slab_layout: &[usize],
    group: &DrawGroup,
    shader_perm: ShaderPermutation,
) -> bool {
    let start = group.instance_range.start as usize;
    let end = group.instance_range.end as usize;
    slab_layout.get(start..end).is_some_and(|members| {
        !members.is_empty()
            && members.iter().all(|&draw_idx| {
                draws
                    .get(draw_idx)
                    .is_some_and(|item| depth_prepass_item_eligible(item, shader_perm))
            })
    })
}

/// Returns whether a draw may be submitted through the conservative generic depth prepass.
fn depth_prepass_item_eligible(item: &WorldMeshDrawItem, shader_perm: ShaderPermutation) -> bool {
    let key = &item.batch_key;
    !item.is_overlay
        && key.render_queue < UNITY_RENDER_QUEUE_ALPHA_TEST
        && !key.alpha_blended
        && !key.blend_mode.is_transparent()
        && !key.embedded_requires_intersection_pass
        && !key.embedded_uses_scene_depth_snapshot
        && !key.embedded_uses_scene_color_snapshot
        && key.render_state.depth_write != Some(false)
        && key.render_state.depth_compare.is_none()
        && key.render_state.depth_offset.is_none()
        && !key.render_state.stencil.enabled
        && match &key.pipeline {
            RasterPipelineKind::Null => true,
            RasterPipelineKind::EmbeddedStem(stem) => {
                embedded_stem_depth_prepass_pass(stem.as_ref(), shader_perm).is_some()
            }
        }
}

/// Mutable output and scratch buffers used while building one [`InstancePlan`].
struct InstancePlanBuilder {
    /// Per-draw slab order emitted for the frame.
    slab_layout: Vec<usize>,
    /// Named phase queues emitted for the frame.
    phases: RenderPhaseSet<WorldMeshPhase, DrawGroup>,
    /// Shader permutation used to decide phase mirrors that depend on material pass metadata.
    shader_perm: ShaderPermutation,
    /// Reusable grouping scratch for one batch-key window.
    scratch: InstancePlanScratch,
}

impl InstancePlanBuilder {
    /// Creates a builder sized for `draw_count` sorted draws.
    fn with_capacity(draw_count: usize, shader_perm: ShaderPermutation) -> Self {
        Self {
            slab_layout: Vec::with_capacity(draw_count),
            phases: RenderPhaseSet::new(),
            shader_perm,
            scratch: InstancePlanScratch::default(),
        }
    }

    /// Emits all groups for one same-batch-key window.
    fn process_window(&mut self, draws: &[WorldMeshDrawItem], window: BatchWindow) {
        if window.singleton {
            self.emit_singletons(draws, window);
        } else {
            self.emit_grouped_window(draws, window);
        }
    }

    /// Emits one GPU draw group per source draw.
    fn emit_singletons(&mut self, draws: &[WorldMeshDrawItem], window: BatchWindow) {
        for draw_idx in window.range {
            let group = build_group(&mut self.slab_layout, draw_idx, &[draw_idx]);
            self.queue_group_to_phase(draws, window.phase, group);
        }
    }

    /// Groups non-transparent same-batch-key draws by mesh/submesh before emission.
    fn emit_grouped_window(&mut self, draws: &[WorldMeshDrawItem], window: BatchWindow) {
        self.scratch.rebuild(draws, window.range.clone());
        for group_idx in 0..self.scratch.group_count() {
            let group = {
                let members = self.scratch.group_members(group_idx);
                let representative = self.scratch.group_representative(group_idx);
                build_group(&mut self.slab_layout, representative, members)
            };
            self.queue_group_to_phase(draws, window.phase, group);
        }
    }

    /// Emits pre-merged groups for a large same-batch-key window.
    fn emit_merged_grouped_window(
        &mut self,
        draws: &[WorldMeshDrawItem],
        phase: WorldMeshPhase,
        groups: Vec<MergedGroupedWindowGroup>,
    ) {
        for merged in groups {
            let group = build_group(
                &mut self.slab_layout,
                merged.representative_draw_idx,
                &merged.members,
            );
            self.queue_group_to_phase(draws, phase, group);
        }
    }

    /// Emits groups already merged by resolved material submission compatibility.
    fn emit_submission_groups(
        &mut self,
        draws: &[WorldMeshDrawItem],
        groups: Vec<PendingSubmissionGroup>,
    ) {
        for pending in groups {
            let group = build_group(
                &mut self.slab_layout,
                pending.representative_draw_idx,
                &pending.members,
            );
            self.queue_group_to_phase(draws, pending.phase, group);
        }
    }

    /// Queues one group into its primary phase and any mirror phases.
    fn queue_group_to_phase(
        &mut self,
        draws: &[WorldMeshDrawItem],
        phase: WorldMeshPhase,
        group: DrawGroup,
    ) {
        self.phases.phase_mut(phase).push(group.clone());
        if phase_is_pre_skybox_forward(phase) {
            self.phases
                .phase_mut(WorldMeshPhase::ViewNormals)
                .push(group.clone());
            if depth_prepass_group_eligible(draws, &self.slab_layout, &group, self.shader_perm) {
                self.phases.phase_mut(WorldMeshPhase::DepthOnly).push(group);
            }
        }
    }

    /// Produces the final plan after debug-validating group order.
    fn finish(self) -> InstancePlan {
        let plan = InstancePlan {
            slab_layout: self.slab_layout,
            phases: self.phases,
        };
        debug_assert_plan_group_order(&plan);
        plan
    }
}

fn phase_is_pre_skybox_forward(phase: WorldMeshPhase) -> bool {
    matches!(
        phase,
        WorldMeshPhase::ForwardOpaque | WorldMeshPhase::ForwardAlphaTest
    )
}

#[cfg(test)]
mod tests;
