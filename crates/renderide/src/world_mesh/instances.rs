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

use rayon::prelude::*;

use crate::materials::{
    RasterPipelineKind, ShaderPermutation, UNITY_RENDER_QUEUE_ALPHA_TEST,
    embedded_stem_depth_prepass_pass, embedded_stem_has_shadow_caster_pass,
    render_queue_is_transparent,
};
#[cfg(test)]
use crate::shared::ShadowCastMode;

use super::draw_prep::WorldMeshDrawItem;

use batch_window::{BatchWindow, emit_group, next_batch_window, subpass_groups};
use scratch::InstancePlanScratch;

/// Draw count above which [`build_plan`] may split batch windows across worker threads.
const INSTANCE_PLAN_PARALLEL_MIN_DRAWS: usize = 1_024;
/// Minimum independent batch windows needed to amortize Rayon scheduling and merge overhead.
const INSTANCE_PLAN_PARALLEL_MIN_WINDOWS: usize = 4;

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

/// Per-view instance plan: slab layout plus groups for pre-skybox regular, post-skybox regular,
/// intersection, and grab-pass transparent subpasses.
///
/// The forward pass packs the per-draw slab in `slab_layout` order -- slot `i` holds the
/// per-draw uniforms for `draws[slab_layout[i]]` -- and emits each group's `instance_range`
/// directly. `representative_draw_idx` for each group list is monotonically increasing; backend
/// frame planning attaches material packet indices after packet resolution so recording does not
/// search packet boundaries.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct InstancePlan {
    /// New slab order. `slab_layout[i]` is the sorted-draw index whose per-draw uniforms
    /// go into per-draw slot `i`. Length equals `draws.len()` (every draw gets one slot).
    pub slab_layout: Vec<usize>,
    /// Groups emitted before the skybox draw, in ascending `representative_draw_idx` order.
    pub regular_groups: Vec<DrawGroup>,
    /// Regular forward groups emitted after the skybox draw, in ascending
    /// `representative_draw_idx` order.
    pub post_skybox_groups: Vec<DrawGroup>,
    /// Groups emitted by the intersection-pass subpass (post depth-snapshot), in
    /// ascending `representative_draw_idx` order.
    pub intersect_groups: Vec<DrawGroup>,
    /// Groups emitted by the grab-pass transparent subpass (post scene-color snapshot), in
    /// ascending `representative_draw_idx` order.
    pub transparent_groups: Vec<DrawGroup>,
}

/// Shadow-map submission path selected for one draw item.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ShadowCasterRoute {
    /// Do not submit this item to shadow maps.
    #[cfg(test)]
    Skip,
    /// Submit through the renderer-owned position-only depth path.
    GenericDepth,
    /// Submit through the material-authored shadow-caster pass.
    MaterialShadow,
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
pub fn build_plan(draws: &[WorldMeshDrawItem], supports_base_instance: bool) -> InstancePlan {
    profiling::scope!("mesh::build_plan");
    if draws.is_empty() {
        return InstancePlan::default();
    }

    if draws.len() < INSTANCE_PLAN_PARALLEL_MIN_DRAWS {
        return build_plan_serial(draws, supports_base_instance);
    }

    let windows = collect_batch_windows(draws, supports_base_instance);
    if should_parallelize_instance_plan(draws.len(), windows.len()) {
        build_plan_parallel(draws, &windows)
    } else {
        build_plan_from_windows_serial(draws, &windows)
    }
}

fn should_parallelize_instance_plan(draw_count: usize, window_count: usize) -> bool {
    draw_count >= INSTANCE_PLAN_PARALLEL_MIN_DRAWS
        && window_count >= INSTANCE_PLAN_PARALLEL_MIN_WINDOWS
}

fn build_plan_serial(draws: &[WorldMeshDrawItem], supports_base_instance: bool) -> InstancePlan {
    let mut builder = InstancePlanBuilder::with_capacity(draws.len());
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
) -> InstancePlan {
    let mut builder = InstancePlanBuilder::with_capacity(draws.len());
    for window in windows {
        builder.process_window(draws, window.clone());
    }
    builder.finish()
}

fn build_plan_parallel(draws: &[WorldMeshDrawItem], windows: &[BatchWindow]) -> InstancePlan {
    profiling::scope!("mesh::build_plan_parallel_windows");
    let partials: Vec<_> = windows
        .par_iter()
        .map(|window| {
            profiling::scope!("mesh::build_plan_window_worker");
            let mut builder = InstancePlanBuilder::with_capacity(window.range.len());
            builder.process_window(draws, window.clone());
            builder.finish()
        })
        .collect();
    merge_partial_instance_plans(draws.len(), partials)
}

fn merge_partial_instance_plans(draw_count: usize, partials: Vec<InstancePlan>) -> InstancePlan {
    let mut plan = InstancePlan {
        slab_layout: Vec::with_capacity(draw_count),
        regular_groups: Vec::new(),
        post_skybox_groups: Vec::new(),
        intersect_groups: Vec::new(),
        transparent_groups: Vec::new(),
    };

    for partial in partials {
        let slab_offset = plan.slab_layout.len() as u32;
        plan.slab_layout.extend(partial.slab_layout);
        append_groups_with_slab_offset(
            &mut plan.regular_groups,
            partial.regular_groups,
            slab_offset,
        );
        append_groups_with_slab_offset(
            &mut plan.post_skybox_groups,
            partial.post_skybox_groups,
            slab_offset,
        );
        append_groups_with_slab_offset(
            &mut plan.intersect_groups,
            partial.intersect_groups,
            slab_offset,
        );
        append_groups_with_slab_offset(
            &mut plan.transparent_groups,
            partial.transparent_groups,
            slab_offset,
        );
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
    debug_assert!(groups_are_monotonic(&plan.regular_groups));
    debug_assert!(groups_are_monotonic(&plan.post_skybox_groups));
    debug_assert!(groups_are_monotonic(&plan.intersect_groups));
    debug_assert!(groups_are_monotonic(&plan.transparent_groups));
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

/// Returns whether a shadow-caster draw group may use the generic position-only depth pass.
#[cfg(test)]
pub(crate) fn shadow_caster_depth_group_eligible(
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
                    .is_some_and(|item| shadow_caster_depth_item_eligible(item, shader_perm))
            })
    })
}

/// Returns whether a draw may be submitted through the conservative generic depth prepass.
fn depth_prepass_item_eligible(item: &WorldMeshDrawItem, shader_perm: ShaderPermutation) -> bool {
    let key = &item.batch_key;
    !item.is_overlay
        && key.render_queue < UNITY_RENDER_QUEUE_ALPHA_TEST
        && !render_queue_is_transparent(key.render_queue)
        && !key.alpha_blended
        && !key.blend_mode.is_transparent()
        && !key.embedded_requires_intersection_pass
        && !key.embedded_uses_scene_depth_snapshot
        && !key.embedded_uses_scene_color_snapshot
        && key.render_state.depth_write != Some(false)
        && key.render_state.depth_compare.is_none()
        && key.render_state.depth_offset.is_none()
        && !key.render_state.stencil.enabled
        && batch_key_has_depth_prepass(item, shader_perm)
}

/// Returns whether a draw can cast into a shadow map through the generic depth-only path.
#[cfg(test)]
fn shadow_caster_depth_item_eligible(
    item: &WorldMeshDrawItem,
    shader_perm: ShaderPermutation,
) -> bool {
    shadow_caster_route_for_item(item, shader_perm) == ShadowCasterRoute::GenericDepth
}

/// Returns whether a draw should cast shadows through its source-authored material pass.
#[cfg(test)]
fn shadow_caster_material_item_eligible(
    item: &WorldMeshDrawItem,
    shader_perm: ShaderPermutation,
) -> bool {
    shadow_caster_route_for_item(item, shader_perm) == ShadowCasterRoute::MaterialShadow
}

/// Selects the cheapest shadow-map path that preserves the material's caster silhouette.
#[cfg(test)]
pub(crate) fn shadow_caster_route_for_item(
    item: &WorldMeshDrawItem,
    shader_perm: ShaderPermutation,
) -> ShadowCasterRoute {
    if item.is_overlay || item.shadow_cast_mode == ShadowCastMode::Off {
        return ShadowCasterRoute::Skip;
    }
    shadow_caster_batch_route_for_item(item, shader_perm)
}

/// Selects the material-dependent shadow-map route for an otherwise shadow-casting draw.
pub(crate) fn shadow_caster_batch_route_for_item(
    item: &WorldMeshDrawItem,
    shader_perm: ShaderPermutation,
) -> ShadowCasterRoute {
    if batch_key_has_shadow_caster_pass(item, shader_perm) {
        ShadowCasterRoute::MaterialShadow
    } else {
        ShadowCasterRoute::GenericDepth
    }
}

/// Returns whether the material route has a position-only depth pipeline.
fn batch_key_has_depth_prepass(item: &WorldMeshDrawItem, shader_perm: ShaderPermutation) -> bool {
    match &item.batch_key.pipeline {
        RasterPipelineKind::Null => true,
        RasterPipelineKind::EmbeddedStem(stem) => {
            embedded_stem_depth_prepass_pass(stem.as_ref(), shader_perm).is_some()
        }
    }
}

/// Returns whether the material route declares a source-authored shadow caster pass.
fn batch_key_has_shadow_caster_pass(
    item: &WorldMeshDrawItem,
    shader_perm: ShaderPermutation,
) -> bool {
    match &item.batch_key.pipeline {
        RasterPipelineKind::Null => false,
        RasterPipelineKind::EmbeddedStem(stem) => {
            embedded_stem_has_shadow_caster_pass(stem.as_ref(), shader_perm)
        }
    }
}

/// Mutable output and scratch buffers used while building one [`InstancePlan`].
struct InstancePlanBuilder {
    /// Per-draw slab order emitted for the frame.
    slab_layout: Vec<usize>,
    /// Regular forward draw groups emitted before the skybox draw.
    regular_groups: Vec<DrawGroup>,
    /// Regular forward draw groups emitted after the skybox draw.
    post_skybox_groups: Vec<DrawGroup>,
    /// Intersection-pass draw groups.
    intersect_groups: Vec<DrawGroup>,
    /// Grab-pass transparent draw groups.
    transparent_groups: Vec<DrawGroup>,
    /// Reusable grouping scratch for one batch-key window.
    scratch: InstancePlanScratch,
}

impl InstancePlanBuilder {
    /// Creates a builder sized for `draw_count` sorted draws.
    fn with_capacity(draw_count: usize) -> Self {
        Self {
            slab_layout: Vec::with_capacity(draw_count),
            regular_groups: Vec::new(),
            post_skybox_groups: Vec::new(),
            intersect_groups: Vec::new(),
            transparent_groups: Vec::new(),
            scratch: InstancePlanScratch::default(),
        }
    }

    /// Emits all groups for one same-batch-key window.
    fn process_window(&mut self, draws: &[WorldMeshDrawItem], window: BatchWindow) {
        if window.singleton {
            self.emit_singletons(window);
        } else {
            self.emit_grouped_window(draws, window);
        }
    }

    /// Emits one GPU draw group per source draw.
    fn emit_singletons(&mut self, window: BatchWindow) {
        let target = subpass_groups(
            &mut self.regular_groups,
            &mut self.post_skybox_groups,
            &mut self.intersect_groups,
            &mut self.transparent_groups,
            &window,
        );
        for draw_idx in window.range {
            emit_group(&mut self.slab_layout, target, draw_idx, &[draw_idx]);
        }
    }

    /// Groups non-transparent same-batch-key draws by mesh/submesh before emission.
    fn emit_grouped_window(&mut self, draws: &[WorldMeshDrawItem], window: BatchWindow) {
        self.scratch.rebuild(draws, window.range.clone());
        let target = subpass_groups(
            &mut self.regular_groups,
            &mut self.post_skybox_groups,
            &mut self.intersect_groups,
            &mut self.transparent_groups,
            &window,
        );
        for group_idx in 0..self.scratch.group_count() {
            let members = self.scratch.group_members(group_idx);
            emit_group(
                &mut self.slab_layout,
                target,
                self.scratch.group_representative(group_idx),
                members,
            );
        }
    }

    /// Produces the final plan after debug-validating group order.
    fn finish(self) -> InstancePlan {
        // The cross-window walk visits regular and intersect groups interleaved by sort order,
        // so each list is already in ascending `representative_draw_idx` order -- no resort.
        debug_assert!(
            self.regular_groups
                .windows(2)
                .all(|w| w[0].representative_draw_idx <= w[1].representative_draw_idx)
        );
        debug_assert!(
            self.intersect_groups
                .windows(2)
                .all(|w| w[0].representative_draw_idx <= w[1].representative_draw_idx)
        );
        debug_assert!(
            self.post_skybox_groups
                .windows(2)
                .all(|w| w[0].representative_draw_idx <= w[1].representative_draw_idx)
        );
        debug_assert!(
            self.transparent_groups
                .windows(2)
                .all(|w| w[0].representative_draw_idx <= w[1].representative_draw_idx)
        );

        InstancePlan {
            slab_layout: self.slab_layout,
            regular_groups: self.regular_groups,
            post_skybox_groups: self.post_skybox_groups,
            intersect_groups: self.intersect_groups,
            transparent_groups: self.transparent_groups,
        }
    }
}

#[cfg(test)]
mod tests;
