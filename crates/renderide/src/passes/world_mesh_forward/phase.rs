//! Prepared draw phases for depth-only world-mesh raster passes.
//!
//! These phases move eligibility checks and depth-pipeline key construction out of render-pass
//! encoding. The encoder only replays compact runs that are already known to belong to that pass.

use std::hash::{Hash, Hasher};

use ahash::AHasher;
use glam::{Mat4, Vec3};

use crate::world_mesh::culling::frustum::world_aabb_visible_in_homogeneous_clip;
use crate::world_mesh::{
    DrawGroup, InstancePlan, ShadowCasterRoute, WorldMeshDrawItem, depth_prepass_group_eligible,
    shadow_caster_batch_route_for_item,
};

use super::WorldMeshForwardPipelineState;
use super::depth_prepass::WorldMeshForwardDepthPrepassPipelineKey;

/// One prepared generic depth draw run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DepthPrepassRun {
    /// Group to submit to the render pass.
    pub(super) group: DrawGroup,
    /// Pipeline key resolved while the phase was built.
    pub(super) pipeline_key: WorldMeshForwardDepthPrepassPipelineKey,
}

/// One prepared material-authored shadow-caster run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MaterialShadowRun {
    /// Group to submit through its material shadow-caster pass.
    pub(super) group: DrawGroup,
}

/// CPU-side summary of phase construction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DrawPhaseBuildStats {
    /// Number of input draw groups inspected by the phase builder.
    pub considered_groups: usize,
    /// Number of groups rejected by pass-specific eligibility rules.
    pub skipped_eligibility_groups: usize,
    /// Number of groups rejected by the shadow-view frustum.
    pub skipped_culled_groups: usize,
    /// Number of groups rejected because a depth pipeline key could not be built.
    pub skipped_pipeline_key_groups: usize,
    /// Number of groups retained in the prepared phase.
    pub prepared_groups: usize,
    /// Number of source instances covered by retained groups.
    pub prepared_instances: usize,
}

/// Prepared safe-opaque depth prepass phase for one forward view.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct DepthPrepassPhase {
    /// Compact replay list for the depth prepass encoder.
    pub runs: Vec<DepthPrepassRun>,
    /// Phase construction diagnostics.
    pub stats: DrawPhaseBuildStats,
}

/// Prepared shadow-map phases for one shadow-caster draw plan.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ShadowCasterPhase {
    /// Compact replay list for generic position-only shadow draws.
    pub depth_runs: Vec<DepthPrepassRun>,
    /// Compact replay list for material-authored shadow-caster draws.
    pub material_runs: Vec<MaterialShadowRun>,
    /// Phase construction diagnostics.
    pub stats: DrawPhaseBuildStats,
}

/// Route and bounds data reused by every shadow view for one prepared caster group.
#[derive(Clone, Debug)]
pub(crate) struct ShadowCasterPreparedPlan {
    /// Prepared caster groups after route classification and mixed-group splitting.
    groups: Vec<PreparedShadowGroup>,
    /// Route-classification diagnostics shared by every shadow view.
    stats: DrawPhaseBuildStats,
    /// Hash of the caster plan and fields that affect view-specific phase replay.
    signature: u64,
}

impl ShadowCasterPreparedPlan {
    /// Returns whether any shadow caster group survived route preparation.
    pub(crate) fn has_work(&self) -> bool {
        !self.groups.is_empty()
    }

    /// Returns whether any prepared group needs a material-authored shadow pass.
    pub(crate) fn has_material_runs(&self) -> bool {
        self.groups
            .iter()
            .any(|group| group.route == PreparedShadowRoute::MaterialShadow)
    }

    /// Returns the stable signature for cached shadow-view phases.
    pub(crate) fn signature(&self) -> u64 {
        self.signature
    }

    /// Assigns resolved material packet indices to prepared material-shadow groups.
    pub(crate) fn assign_material_packet_indices(&mut self, packet_last_draw_indices: &[usize]) {
        if packet_last_draw_indices.is_empty() {
            return;
        }
        let mut packet_idx = 0usize;
        for group in &mut self.groups {
            if group.route != PreparedShadowRoute::MaterialShadow {
                continue;
            }
            let representative = group.group.representative_draw_idx;
            while packet_idx + 1 < packet_last_draw_indices.len()
                && packet_last_draw_indices[packet_idx] < representative
            {
                packet_idx += 1;
            }
            group.group.material_packet_idx = packet_idx;
        }
        self.signature = compute_shadow_caster_plan_signature(self);
    }
}

/// One prepared caster group with a fixed submission route.
#[derive(Clone, Debug)]
struct PreparedShadowGroup {
    /// Draw group to replay if the shadow view keeps this caster.
    group: DrawGroup,
    /// Shadow-map route selected for this group.
    route: PreparedShadowRoute,
    /// Conservative world-space bounds used for light-frustum rejection.
    bounds: PreparedShadowBounds,
    /// Hash of source slab members covered by this group.
    member_signature: u64,
}

/// Shadow-map route selected during prepared-plan construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum PreparedShadowRoute {
    /// Submit through the renderer-owned position-only depth path.
    GenericDepth(WorldMeshForwardDepthPrepassPipelineKey),
    /// Submit through the material-authored shadow-caster pass.
    MaterialShadow,
}

/// Conservative world-space bounds for one prepared caster group.
#[derive(Clone, Copy, Debug)]
enum PreparedShadowBounds {
    /// At least one source draw had missing or non-finite bounds, so the group must be kept.
    Unknown,
    /// Finite union bounds for all members in the prepared group.
    Finite {
        /// Minimum world-space corner.
        min: Vec3,
        /// Maximum world-space corner.
        max: Vec3,
    },
}

/// Builds the safe-opaque depth prepass phase for one prepared forward view.
pub(crate) fn build_depth_prepass_phase(
    draws: &[WorldMeshDrawItem],
    plan: &InstancePlan,
    pipeline: &WorldMeshForwardPipelineState,
) -> DepthPrepassPhase {
    profiling::scope!("world_mesh::build_depth_prepass_phase");
    let mut phase = DepthPrepassPhase {
        runs: Vec::with_capacity(plan.regular_groups.len()),
        stats: DrawPhaseBuildStats::default(),
    };
    for group in &plan.regular_groups {
        push_depth_run_if_eligible(
            &mut phase.runs,
            &mut phase.stats,
            draws,
            &plan.slab_layout,
            group,
            pipeline,
        );
    }
    phase
}

/// Builds the shadow-map phases for one shadow-caster draw plan and optional shadow view.
#[cfg(test)]
pub(crate) fn build_shadow_caster_phase_for_view(
    draws: &[WorldMeshDrawItem],
    plan: &InstancePlan,
    pipeline: &WorldMeshForwardPipelineState,
    shadow_view_proj: Option<Mat4>,
) -> ShadowCasterPhase {
    let prepared = build_shadow_caster_prepared_plan(draws, plan, pipeline);
    build_shadow_caster_phase_from_prepared(&prepared, shadow_view_proj)
}

/// Builds route-classified caster groups that can be reused by all shadow-map views.
pub(crate) fn build_shadow_caster_prepared_plan(
    draws: &[WorldMeshDrawItem],
    plan: &InstancePlan,
    pipeline: &WorldMeshForwardPipelineState,
) -> ShadowCasterPreparedPlan {
    profiling::scope!("world_mesh::build_shadow_caster_prepared_plan");
    let group_count = plan
        .regular_groups
        .len()
        .saturating_add(plan.post_skybox_groups.len())
        .saturating_add(plan.intersect_groups.len())
        .saturating_add(plan.transparent_groups.len());
    let mut builder = ShadowCasterPreparedPlanBuilder {
        groups: Vec::with_capacity(group_count),
        stats: DrawPhaseBuildStats::default(),
    };
    for groups in [
        plan.regular_groups.as_slice(),
        plan.post_skybox_groups.as_slice(),
        plan.intersect_groups.as_slice(),
        plan.transparent_groups.as_slice(),
    ] {
        builder.append_group_slice(draws, &plan.slab_layout, pipeline, groups);
    }
    let mut prepared = builder.finish();
    prepared.signature = compute_shadow_caster_plan_signature(&prepared);
    prepared
}

/// Builds the shadow-map phase for one view from a retained prepared caster plan.
pub(crate) fn build_shadow_caster_phase_from_prepared(
    prepared: &ShadowCasterPreparedPlan,
    shadow_view_proj: Option<Mat4>,
) -> ShadowCasterPhase {
    profiling::scope!("world_mesh::build_shadow_caster_phase_from_prepared");
    let mut phase = ShadowCasterPhase {
        depth_runs: Vec::with_capacity(prepared.groups.len()),
        material_runs: Vec::new(),
        stats: prepared.stats,
    };
    for group in &prepared.groups {
        if !group.bounds.visible(shadow_view_proj) {
            phase.stats.skipped_culled_groups = phase.stats.skipped_culled_groups.saturating_add(1);
            continue;
        }
        match group.route {
            PreparedShadowRoute::MaterialShadow => {
                push_material_shadow_run(&mut phase, &group.group);
            }
            PreparedShadowRoute::GenericDepth(pipeline_key) => {
                push_depth_run(&mut phase, &group.group, pipeline_key);
            }
        }
    }
    phase
}

/// Incremental builder for a prepared shadow-caster plan.
struct ShadowCasterPreparedPlanBuilder {
    /// Prepared caster groups.
    groups: Vec<PreparedShadowGroup>,
    /// Route preparation diagnostics.
    stats: DrawPhaseBuildStats,
}

impl ShadowCasterPreparedPlanBuilder {
    /// Appends prepared shadow runs for one instance-plan group slice.
    fn append_group_slice(
        &mut self,
        draws: &[WorldMeshDrawItem],
        slab_layout: &[usize],
        pipeline: &WorldMeshForwardPipelineState,
        groups: &[DrawGroup],
    ) {
        self.groups.reserve(groups.len());
        for group in groups {
            self.append_group(draws, slab_layout, pipeline, group);
        }
    }

    /// Appends one source draw group, splitting it when individual members need different routes.
    fn append_group(
        &mut self,
        draws: &[WorldMeshDrawItem],
        slab_layout: &[usize],
        pipeline: &WorldMeshForwardPipelineState,
        group: &DrawGroup,
    ) {
        self.stats.considered_groups = self.stats.considered_groups.saturating_add(1);
        let start = group.instance_range.start as usize;
        let end = group.instance_range.end as usize;
        let Some(members) = slab_layout.get(start..end) else {
            self.stats.skipped_eligibility_groups =
                self.stats.skipped_eligibility_groups.saturating_add(1);
            return;
        };
        if members.is_empty() {
            self.stats.skipped_eligibility_groups =
                self.stats.skipped_eligibility_groups.saturating_add(1);
            return;
        }
        let Some(batch_route) = members
            .iter()
            .find_map(|&draw_idx| draws.get(draw_idx))
            .map(|item| shadow_caster_batch_route_for_item(item, pipeline.shader_perm))
        else {
            self.stats.skipped_eligibility_groups =
                self.stats.skipped_eligibility_groups.saturating_add(1);
            return;
        };

        let mut run: Option<PreparedShadowRunBuilder> = None;
        let mut generic_pipeline_key = CachedGenericShadowPipelineKey::Unresolved;
        let mut prepared_any = false;
        let mut skipped_for_pipeline = false;
        for (member_offset, &draw_idx) in members.iter().enumerate() {
            let slab_slot = start.saturating_add(member_offset);
            let Some(item) = draws.get(draw_idx) else {
                flush_prepared_shadow_run(&mut self.groups, &mut run, draws, slab_layout, group);
                continue;
            };
            let Some(route) = prepared_shadow_route_for_item(
                item,
                pipeline,
                batch_route,
                &mut generic_pipeline_key,
            ) else {
                flush_prepared_shadow_run(&mut self.groups, &mut run, draws, slab_layout, group);
                if shadow_item_can_cast(item) && batch_route == ShadowCasterRoute::GenericDepth {
                    skipped_for_pipeline = true;
                }
                continue;
            };
            if run.as_ref().is_some_and(|active| active.route != route) {
                flush_prepared_shadow_run(&mut self.groups, &mut run, draws, slab_layout, group);
            }
            if let Some(active) = run.as_mut() {
                active.end_slot = slab_slot.saturating_add(1);
            } else {
                run = Some(PreparedShadowRunBuilder {
                    route,
                    start_slot: slab_slot,
                    end_slot: slab_slot.saturating_add(1),
                    representative_draw_idx: draw_idx,
                });
            }
            prepared_any = true;
        }
        flush_prepared_shadow_run(&mut self.groups, &mut run, draws, slab_layout, group);
        if !prepared_any {
            if skipped_for_pipeline {
                self.stats.skipped_pipeline_key_groups =
                    self.stats.skipped_pipeline_key_groups.saturating_add(1);
            } else {
                self.stats.skipped_eligibility_groups =
                    self.stats.skipped_eligibility_groups.saturating_add(1);
            }
        }
    }

    /// Finalizes the prepared plan.
    fn finish(self) -> ShadowCasterPreparedPlan {
        ShadowCasterPreparedPlan {
            groups: self.groups,
            stats: self.stats,
            signature: 0,
        }
    }
}

/// Mutable state for one contiguous prepared caster run.
struct PreparedShadowRunBuilder {
    /// Submission route for the run.
    route: PreparedShadowRoute,
    /// First source slab slot covered by the run.
    start_slot: usize,
    /// Exclusive source slab slot covered by the run.
    end_slot: usize,
    /// Representative sorted-draw index for material and mesh state.
    representative_draw_idx: usize,
}

/// Cached generic-depth shadow pipeline route for one source draw group.
enum CachedGenericShadowPipelineKey {
    /// No member has requested the generic-depth route yet.
    Unresolved,
    /// Generic-depth routing is unavailable for this source draw group.
    Unavailable,
    /// Generic-depth routing resolved to this pipeline key.
    Available(WorldMeshForwardDepthPrepassPipelineKey),
}

impl CachedGenericShadowPipelineKey {
    /// Resolves the generic-depth pipeline key once and reuses it for later group members.
    fn resolve(
        &mut self,
        item: &WorldMeshDrawItem,
        pipeline: &WorldMeshForwardPipelineState,
    ) -> Option<WorldMeshForwardDepthPrepassPipelineKey> {
        match *self {
            Self::Unresolved => {
                let Some(pipeline_key) =
                    WorldMeshForwardDepthPrepassPipelineKey::for_shadow_draw(item, pipeline)
                else {
                    *self = Self::Unavailable;
                    return None;
                };
                *self = Self::Available(pipeline_key);
                Some(pipeline_key)
            }
            Self::Unavailable => None,
            Self::Available(pipeline_key) => Some(pipeline_key),
        }
    }
}

/// Converts one run builder into a retained prepared caster group.
fn flush_prepared_shadow_run(
    groups: &mut Vec<PreparedShadowGroup>,
    run: &mut Option<PreparedShadowRunBuilder>,
    draws: &[WorldMeshDrawItem],
    slab_layout: &[usize],
    source_group: &DrawGroup,
) {
    let Some(active) = run.take() else {
        return;
    };
    let Some(start) = u32::try_from(active.start_slot).ok() else {
        return;
    };
    let Some(end) = u32::try_from(active.end_slot).ok() else {
        return;
    };
    let mut group = source_group.clone();
    group.representative_draw_idx = active.representative_draw_idx;
    group.instance_range = start..end;
    groups.push(PreparedShadowGroup {
        bounds: prepared_shadow_bounds(draws, slab_layout, active.start_slot, active.end_slot),
        member_signature: prepared_shadow_member_signature(
            draws,
            slab_layout,
            active.start_slot,
            active.end_slot,
        ),
        route: active.route,
        group,
    });
}

/// Selects the prepared caster route for one item.
fn prepared_shadow_route_for_item(
    item: &WorldMeshDrawItem,
    pipeline: &WorldMeshForwardPipelineState,
    batch_route: ShadowCasterRoute,
    generic_pipeline_key: &mut CachedGenericShadowPipelineKey,
) -> Option<PreparedShadowRoute> {
    if !shadow_item_can_cast(item) {
        return None;
    }
    match batch_route {
        #[cfg(test)]
        ShadowCasterRoute::Skip => None,
        ShadowCasterRoute::MaterialShadow => Some(PreparedShadowRoute::MaterialShadow),
        ShadowCasterRoute::GenericDepth => {
            let pipeline_key = generic_pipeline_key.resolve(item, pipeline)?;
            Some(PreparedShadowRoute::GenericDepth(pipeline_key))
        }
    }
}

/// Returns whether a draw is allowed to cast shadows before material routing is considered.
fn shadow_item_can_cast(item: &WorldMeshDrawItem) -> bool {
    !item.is_overlay && item.shadow_cast_mode != crate::shared::ShadowCastMode::Off
}

/// Computes conservative union bounds for a prepared caster group.
fn prepared_shadow_bounds(
    draws: &[WorldMeshDrawItem],
    slab_layout: &[usize],
    start: usize,
    end: usize,
) -> PreparedShadowBounds {
    let Some(members) = slab_layout.get(start..end) else {
        return PreparedShadowBounds::Unknown;
    };
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for &draw_idx in members {
        let Some(item) = draws.get(draw_idx) else {
            return PreparedShadowBounds::Unknown;
        };
        let Some((world_min, world_max)) = item.world_aabb else {
            return PreparedShadowBounds::Unknown;
        };
        if !(world_min.is_finite() && world_max.is_finite()) {
            return PreparedShadowBounds::Unknown;
        }
        min = min.min(world_min.min(world_max));
        max = max.max(world_min.max(world_max));
    }
    if min.is_finite() && max.is_finite() {
        PreparedShadowBounds::Finite { min, max }
    } else {
        PreparedShadowBounds::Unknown
    }
}

/// Hashes the source slab members that will be copied into a compact shadow phase.
fn prepared_shadow_member_signature(
    draws: &[WorldMeshDrawItem],
    slab_layout: &[usize],
    start: usize,
    end: usize,
) -> u64 {
    let mut hasher = AHasher::default();
    let Some(members) = slab_layout.get(start..end) else {
        return 0;
    };
    for &draw_idx in members {
        draw_idx.hash(&mut hasher);
        if let Some(item) = draws.get(draw_idx) {
            item.space_id.hash(&mut hasher);
            item.node_id.hash(&mut hasher);
            item.renderable_index.hash(&mut hasher);
            item.instance_id.hash(&mut hasher);
            item.mesh_asset_id.hash(&mut hasher);
            item.slot_index.hash(&mut hasher);
            item.first_index.hash(&mut hasher);
            item.index_count.hash(&mut hasher);
            item.is_overlay.hash(&mut hasher);
            (item.shadow_cast_mode as u8).hash(&mut hasher);
            item.batch_key_hash.hash(&mut hasher);
            hash_optional_aabb_bits(&mut hasher, item.world_aabb);
        }
    }
    hasher.finish()
}

impl PreparedShadowBounds {
    /// Returns whether these bounds should be retained for an optional shadow view.
    fn visible(self, shadow_view_proj: Option<Mat4>) -> bool {
        let Some(shadow_view_proj) = shadow_view_proj else {
            return true;
        };
        match self {
            Self::Unknown => true,
            Self::Finite { min, max } => {
                world_aabb_visible_in_homogeneous_clip(shadow_view_proj, min, max)
            }
        }
    }
}

/// Adds one material-shadow run and updates phase diagnostics.
fn push_material_shadow_run(phase: &mut ShadowCasterPhase, group: &DrawGroup) {
    phase.material_runs.push(MaterialShadowRun {
        group: group.clone(),
    });
    phase.stats.prepared_groups = phase.stats.prepared_groups.saturating_add(1);
    phase.stats.prepared_instances = phase
        .stats
        .prepared_instances
        .saturating_add(group.instance_range.len());
}

/// Adds one generic depth-shadow run and updates phase diagnostics.
fn push_depth_run(
    phase: &mut ShadowCasterPhase,
    group: &DrawGroup,
    pipeline_key: WorldMeshForwardDepthPrepassPipelineKey,
) {
    phase.depth_runs.push(DepthPrepassRun {
        group: group.clone(),
        pipeline_key,
    });
    phase.stats.prepared_groups = phase.stats.prepared_groups.saturating_add(1);
    phase.stats.prepared_instances = phase
        .stats
        .prepared_instances
        .saturating_add(group.instance_range.len());
}

/// Adds a depth-prepass run when the group is valid for the selected depth pass.
fn push_depth_run_if_eligible(
    runs: &mut Vec<DepthPrepassRun>,
    stats: &mut DrawPhaseBuildStats,
    draws: &[WorldMeshDrawItem],
    slab_layout: &[usize],
    group: &DrawGroup,
    pipeline: &WorldMeshForwardPipelineState,
) {
    stats.considered_groups = stats.considered_groups.saturating_add(1);
    if !depth_prepass_group_eligible(draws, slab_layout, group, pipeline.shader_perm) {
        stats.skipped_eligibility_groups = stats.skipped_eligibility_groups.saturating_add(1);
        return;
    }
    let representative = group.representative_draw_idx;
    let Some(item) = draws.get(representative) else {
        stats.skipped_pipeline_key_groups = stats.skipped_pipeline_key_groups.saturating_add(1);
        return;
    };
    let pipeline_key = WorldMeshForwardDepthPrepassPipelineKey::for_draw(item, pipeline);
    let Some(pipeline_key) = pipeline_key else {
        stats.skipped_pipeline_key_groups = stats.skipped_pipeline_key_groups.saturating_add(1);
        return;
    };
    runs.push(DepthPrepassRun {
        group: group.clone(),
        pipeline_key,
    });
    stats.prepared_groups = stats.prepared_groups.saturating_add(1);
    stats.prepared_instances = stats
        .prepared_instances
        .saturating_add(group.instance_range.len());
}

/// Computes a stable signature for fields that affect prepared shadow phase replay.
fn compute_shadow_caster_plan_signature(prepared: &ShadowCasterPreparedPlan) -> u64 {
    let mut hasher = AHasher::default();
    prepared.stats.considered_groups.hash(&mut hasher);
    prepared.stats.skipped_eligibility_groups.hash(&mut hasher);
    prepared.stats.skipped_pipeline_key_groups.hash(&mut hasher);
    prepared.groups.len().hash(&mut hasher);
    for group in &prepared.groups {
        group.group.representative_draw_idx.hash(&mut hasher);
        group.group.instance_range.start.hash(&mut hasher);
        group.group.instance_range.end.hash(&mut hasher);
        group.group.material_packet_idx.hash(&mut hasher);
        group.route.hash(&mut hasher);
        group.bounds.hash_bits(&mut hasher);
        group.member_signature.hash(&mut hasher);
    }
    hasher.finish()
}

impl PreparedShadowBounds {
    /// Hashes floating-point bounds by their exact bit representation.
    fn hash_bits(self, hasher: &mut impl Hasher) {
        match self {
            Self::Unknown => {
                0u8.hash(hasher);
            }
            Self::Finite { min, max } => {
                1u8.hash(hasher);
                hash_vec3_bits(hasher, min);
                hash_vec3_bits(hasher, max);
            }
        }
    }
}

/// Hashes one vector by exact component bits.
fn hash_vec3_bits(hasher: &mut impl Hasher, value: Vec3) {
    value.x.to_bits().hash(hasher);
    value.y.to_bits().hash(hasher);
    value.z.to_bits().hash(hasher);
}

/// Hashes optional world bounds by exact component bits.
fn hash_optional_aabb_bits(hasher: &mut impl Hasher, value: Option<(Vec3, Vec3)>) {
    match value {
        Some((min, max)) => {
            1u8.hash(hasher);
            hash_vec3_bits(hasher, min);
            hash_vec3_bits(hasher, max);
        }
        None => {
            0u8.hash(hasher);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::materials::{
        MaterialPipelineDesc, MaterialPipelineTarget, RasterPrimitiveTopology, ShaderPermutation,
    };
    use crate::passes::world_mesh_forward::WorldMeshForwardPipelineState;
    use crate::shared::ShadowCastMode;
    use crate::world_mesh::test_fixtures::{DummyDrawItemSpec, dummy_world_mesh_draw_item};

    use super::*;

    fn pipeline_state(target: MaterialPipelineTarget) -> WorldMeshForwardPipelineState {
        WorldMeshForwardPipelineState {
            use_multiview: false,
            pass_desc: MaterialPipelineDesc {
                target,
                surface_format: wgpu::TextureFormat::Rgba16Float,
                depth_stencil_format: Some(wgpu::TextureFormat::Depth24PlusStencil8),
                sample_count: 1,
                multiview_mask: None,
            },
            shader_perm: ShaderPermutation(0),
        }
    }

    fn draw_group(draw_idx: usize) -> DrawGroup {
        DrawGroup {
            representative_draw_idx: draw_idx,
            instance_range: draw_idx as u32..draw_idx as u32 + 1,
            material_packet_idx: 0,
        }
    }

    fn base_draw() -> WorldMeshDrawItem {
        let mut item = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 1,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id: 0,
            slot_index: 0,
            collect_order: 0,
            alpha_blended: false,
        });
        item.batch_key.primitive_topology = RasterPrimitiveTopology::TriangleList;
        item
    }

    /// Returns clip-space-visible world bounds for identity view-projection tests.
    fn visible_bounds() -> (Vec3, Vec3) {
        (Vec3::new(-0.25, -0.25, 0.1), Vec3::new(0.25, 0.25, 0.5))
    }

    /// Returns clip-space-culled world bounds for identity view-projection tests.
    fn culled_bounds() -> (Vec3, Vec3) {
        (Vec3::new(4.0, 4.0, 4.0), Vec3::new(5.0, 5.0, 5.0))
    }

    #[test]
    fn depth_prepass_phase_keeps_only_eligible_groups() {
        let mut eligible = base_draw();
        eligible.shadow_cast_mode = ShadowCastMode::On;
        let mut overlay = base_draw();
        overlay.is_overlay = true;
        let draws = vec![eligible, overlay];
        let plan = InstancePlan {
            slab_layout: vec![0, 1],
            regular_groups: vec![draw_group(0), draw_group(1)],
            post_skybox_groups: Vec::new(),
            intersect_groups: Vec::new(),
            transparent_groups: Vec::new(),
        };

        let phase = build_depth_prepass_phase(
            &draws,
            &plan,
            &pipeline_state(MaterialPipelineTarget::Color),
        );

        assert_eq!(phase.runs.len(), 1);
        assert_eq!(phase.runs[0].group.representative_draw_idx, 0);
        assert_eq!(phase.stats.considered_groups, 2);
        assert_eq!(phase.stats.skipped_eligibility_groups, 1);
        assert_eq!(phase.stats.prepared_groups, 1);
    }

    #[test]
    fn shadow_phase_keeps_generic_depth_groups_without_material_passes() {
        let mut caster = base_draw();
        caster.shadow_cast_mode = ShadowCastMode::On;
        let mut disabled = base_draw();
        disabled.shadow_cast_mode = ShadowCastMode::Off;
        let draws = vec![caster, disabled];
        let plan = InstancePlan {
            slab_layout: vec![0, 1],
            regular_groups: vec![draw_group(0), draw_group(1)],
            post_skybox_groups: Vec::new(),
            intersect_groups: Vec::new(),
            transparent_groups: Vec::new(),
        };

        let phase = build_shadow_caster_phase_for_view(
            &draws,
            &plan,
            &pipeline_state(MaterialPipelineTarget::ShadowCaster),
            None,
        );

        assert_eq!(phase.depth_runs.len(), 1);
        assert_eq!(phase.material_runs.len(), 0);
        assert_eq!(phase.depth_runs[0].group.representative_draw_idx, 0);
        assert_eq!(phase.stats.considered_groups, 2);
        assert_eq!(phase.stats.skipped_eligibility_groups, 1);
        assert_eq!(phase.stats.prepared_groups, 1);
    }

    #[test]
    fn shadow_phase_culls_groups_outside_shadow_view() {
        let mut visible = base_draw();
        visible.shadow_cast_mode = ShadowCastMode::On;
        visible.world_aabb = Some(visible_bounds());
        let mut culled = base_draw();
        culled.shadow_cast_mode = ShadowCastMode::On;
        culled.world_aabb = Some(culled_bounds());
        let draws = vec![visible, culled];
        let plan = InstancePlan {
            slab_layout: vec![0, 1],
            regular_groups: vec![draw_group(0), draw_group(1)],
            post_skybox_groups: Vec::new(),
            intersect_groups: Vec::new(),
            transparent_groups: Vec::new(),
        };

        let phase = build_shadow_caster_phase_for_view(
            &draws,
            &plan,
            &pipeline_state(MaterialPipelineTarget::ShadowCaster),
            Some(Mat4::IDENTITY),
        );

        assert_eq!(phase.depth_runs.len(), 1);
        assert_eq!(phase.depth_runs[0].group.representative_draw_idx, 0);
        assert_eq!(phase.stats.skipped_culled_groups, 1);
        assert_eq!(phase.stats.prepared_groups, 1);
    }
}
