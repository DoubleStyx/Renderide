//! Batch and draw counters for the debug HUD (aligned with sorted [`WorldMeshDrawItem`] order).

use super::draw_prep::{WorldMeshDrawArrangementStats, WorldMeshDrawItem};
use super::instances::DrawGroup;
use super::materials::{MaterialDrawBatchKey, TransparentMaterialClass};
use crate::materials::{
    MaterialBlendMode, MaterialDepthCompareOverride, RasterPipelineKind, ShaderPermutation,
    embedded_stem_pipeline_pass_count,
};

/// Per-class transparent draw counts for diagnostics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WorldMeshTransparentClassStats {
    /// Normal order-dependent alpha-blended draws.
    pub ordered_alpha_draws: usize,
    /// Transparent draws whose effective pass writes depth.
    pub depth_writing_draws: usize,
    /// Grab-pass or scene-color filter draws.
    pub grab_pass_filter_draws: usize,
    /// Additive or multiplicative draws that can relax batching inside their bucket.
    pub commutative_blend_draws: usize,
    /// Known two-sided transparent draws with authored front/back pass order.
    pub known_two_sided_draws: usize,
    /// Transparent draws kept on conservative compatibility ordering.
    pub compatibility_fallback_draws: usize,
}

impl WorldMeshTransparentClassStats {
    /// Adds one draw from `class` to the matching counter.
    fn add(&mut self, class: TransparentMaterialClass) {
        match class {
            TransparentMaterialClass::Opaque => {}
            TransparentMaterialClass::OrderedAlpha => self.ordered_alpha_draws += 1,
            TransparentMaterialClass::DepthWritingTransparent => self.depth_writing_draws += 1,
            TransparentMaterialClass::GrabPassFilter => self.grab_pass_filter_draws += 1,
            TransparentMaterialClass::CommutativeBlend => self.commutative_blend_draws += 1,
            TransparentMaterialClass::KnownTwoSidedTransparent => self.known_two_sided_draws += 1,
            TransparentMaterialClass::CompatibilityFallback => {
                self.compatibility_fallback_draws += 1;
            }
        }
    }
}

/// Draw and batch counts for the debug HUD (aligned with sorted [`WorldMeshDrawItem`] order).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WorldMeshDrawStats {
    /// Distinct `(batch_key, overlay)` groups after sorting.
    pub batch_total: usize,
    /// Batches for non-overlay draws only.
    pub batch_main: usize,
    /// Batches for overlay draws only.
    pub batch_overlay: usize,
    /// Total indexed draws submitted.
    pub draws_total: usize,
    /// Draws in the main (non-overlay) layer.
    pub draws_main: usize,
    /// Draws in the overlay layer.
    pub draws_overlay: usize,
    /// Non-skinned mesh draws.
    pub rigid_draws: usize,
    /// Skinned mesh draws.
    pub skinned_draws: usize,
    /// Slots that went through frustum culling before the final draw list (if culling was enabled).
    pub draws_pre_cull: usize,
    /// Draws removed by frustum culling.
    pub draws_culled: usize,
    /// Draws removed by Hi-Z occlusion when enabled.
    pub draws_hi_z_culled: usize,
    /// GPU instance batches after merge (one indexed draw each); at most `draws_total`.
    ///
    /// Counts batches across visible forward phases, matching what `draw_subset` submits per frame
    /// rather than the optimistic single-pass count.
    pub instance_batch_total: usize,
    /// Subset of [`Self::instance_batch_total`] in the intersection-pass subpass
    /// (materials whose embedded shader needs `_IntersectColor` / depth snapshot).
    ///
    /// Surfaced so the HUD shows how much of the batch count comes from the partition
    /// that the regular opaque subpass cannot merge across.
    pub intersect_pass_batches: usize,
    /// Subset of [`Self::instance_batch_total`] in the grab-pass transparent subpass
    /// (materials whose embedded shader samples the scene-color snapshot).
    pub transparent_pass_batches: usize,
    /// Transparent draw counts by renderer-local material behavior class.
    pub transparent_class_stats: WorldMeshTransparentClassStats,
    /// Opaque indexed batches mirrored by the generic depth prepass.
    pub depth_prepass_batches: usize,
    /// Opaque GPU instances mirrored by the generic depth prepass.
    pub depth_prepass_instances: usize,
    /// Sum of `instance_count` across all emitted batches.
    ///
    /// Equals [`Self::draws_total`] today (every sorted draw is emitted exactly once);
    /// the per-batch instance count reveals how much the merge actually compressed the
    /// submission stream. Compression ratio = `gpu_instances_emitted / instance_batch_total`.
    pub gpu_instances_emitted: usize,
    /// Actual pipeline-pass draw submissions after multi-pass materials expand each instance batch.
    pub submitted_pipeline_pass_total: usize,
    /// Nontransparent bins emitted during CPU draw arrangement.
    pub nontransparent_bins: usize,
    /// Draws emitted through nontransparent phase bins.
    pub nontransparent_binned_draws: usize,
    /// Draws that kept strict transparent/grab sorting.
    pub strict_sorted_draws: usize,
}

/// One submitted draw row for the **Draw state** debug HUD tab.
///
/// Rows are captured after culling and sorting, so `draw_index` matches the per-draw slab index used
/// by `draw_indexed(..., first_instance)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorldMeshDrawStateRow {
    /// Index in the sorted draw list and GPU per-draw slab.
    pub draw_index: usize,
    /// Host render space id.
    pub space_id: i32,
    /// Scene graph node id.
    pub node_id: i32,
    /// Resident mesh asset id.
    pub mesh_asset_id: i32,
    /// Renderer material slot index.
    pub slot_index: usize,
    /// Host shader asset id from material `set_shader`.
    pub shader_asset_id: i32,
    /// Material asset id.
    pub material_asset_id: i32,
    /// Slot0 property block id when present.
    pub property_block_slot0: Option<i32>,
    /// Resolved raster pipeline route.
    pub pipeline: RasterPipelineKind,
    /// Resolved material blend override used for pipeline selection.
    pub blend_mode: MaterialBlendMode,
    /// True for overlay-layer draws.
    pub is_overlay: bool,
    /// Host sorting order carried by the mesh renderer.
    pub sorting_order: i32,
    /// Stable collection order before sort.
    pub collect_order: usize,
    /// Whether the draw uses the skinned/deformed vertex path.
    pub skinned: bool,
    /// Whether this draw is alpha sorted.
    pub alpha_blended: bool,
    /// Renderer-local transparent behavior class.
    pub transparent_class: TransparentMaterialClass,
    /// Whether this draw is emitted through the intersection depth-snapshot subpass.
    pub requires_intersection_pass: bool,
    /// Unity `_ZWrite` / `ZWrite` override. `None` means the shader pass default is used.
    pub depth_write: Option<bool>,
    /// Material depth-compare override. `None` means the shader pass default is used.
    pub depth_compare: Option<MaterialDepthCompareOverride>,
    /// Unity `Offset factor, units` override. `None` means the shader pass default is used.
    pub depth_offset: Option<(u32, i32)>,
    /// Whether stencil state was enabled by material/properties.
    pub stencil_enabled: bool,
    /// Dynamic stencil reference.
    pub stencil_reference: u32,
    /// Unity `CompareFunction` enum value.
    pub stencil_compare: u8,
    /// Unity `StencilOp` enum value.
    pub stencil_pass_op: u8,
    /// Stencil read mask.
    pub stencil_read_mask: u32,
    /// Stencil write mask.
    pub stencil_write_mask: u32,
    /// Unity `_ColorMask` / `ColorMask` override. `None` means the shader pass default is used.
    pub color_mask: Option<u8>,
}

/// Computes batch boundaries from material/property-block/skin/overlay changes after sorting.
///
/// `supports_base_instance` should match the forward pass (see [`crate::passes::WorldMeshForwardOpaquePass`])
/// so [`WorldMeshDrawStats::instance_batch_total`] reflects the same merge policy.
pub fn stats_from_sorted(
    draws: &[WorldMeshDrawItem],
    cull: Option<(usize, usize, usize)>,
    arrangement: WorldMeshDrawArrangementStats,
    supports_base_instance: bool,
    shader_perm: ShaderPermutation,
) -> WorldMeshDrawStats {
    let draws_total = draws.len();
    let draws_main = draws.iter().filter(|d| !d.is_overlay).count();
    let draws_overlay = draws_total - draws_main;
    let rigid_draws = draws.iter().filter(|d| !d.skinned).count();
    let skinned_draws = draws_total - rigid_draws;
    let transparent_class_stats = transparent_class_stats_from_sorted(draws);

    let mut batch_total = 0usize;
    let mut batch_main = 0usize;
    let mut batch_overlay = 0usize;
    let mut prev: Option<(MaterialDrawBatchKey, bool)> = None;
    for d in draws {
        let cur = (d.batch_key.clone(), d.is_overlay);
        let same_as_prev = prev
            .as_ref()
            .is_some_and(|(k, o)| k == &d.batch_key && *o == d.is_overlay);
        if !same_as_prev {
            batch_total += 1;
            if d.is_overlay {
                batch_overlay += 1;
            } else {
                batch_main += 1;
            }
            prev = Some(cur);
        }
    }

    let (draws_pre_cull, draws_culled, draws_hi_z_culled) = cull.unwrap_or((0, 0, 0));

    // The forward pass drives both subpasses from this same `InstancePlan`, so the HUD
    // counts are exactly what `draw_subset` ends up submitting.
    let plan = crate::world_mesh::build_plan_for_shader(draws, supports_base_instance, shader_perm);
    let intersect_pass_batches = plan.phase_len(crate::world_mesh::WorldMeshPhase::Intersection);
    let transparent_pass_batches = plan.phase_len(crate::world_mesh::WorldMeshPhase::Transparent)
        + plan.phase_len(crate::world_mesh::WorldMeshPhase::TransparentGrab);
    let (depth_prepass_batches, depth_prepass_instances) =
        depth_prepass_counts(plan.phase(crate::world_mesh::WorldMeshPhase::DepthOnly));
    let instance_batch_total = plan.primary_forward_group_count();
    let gpu_instances_emitted: usize = plan
        .primary_forward_groups()
        .map(|g| (g.instance_range.end - g.instance_range.start) as usize)
        .sum();
    // this is the real submit count when a material has multiple passes.
    let submitted_pipeline_pass_total = plan
        .primary_forward_groups()
        .map(|group: &DrawGroup| {
            let item = &draws[group.representative_draw_idx];
            match &item.batch_key.pipeline {
                RasterPipelineKind::EmbeddedStem(stem) => {
                    embedded_stem_pipeline_pass_count(stem.as_ref(), shader_perm)
                }
                RasterPipelineKind::Null => 1,
            }
        })
        .sum();

    WorldMeshDrawStats {
        batch_total,
        batch_main,
        batch_overlay,
        draws_total,
        draws_main,
        draws_overlay,
        rigid_draws,
        skinned_draws,
        draws_pre_cull,
        draws_culled,
        draws_hi_z_culled,
        instance_batch_total,
        intersect_pass_batches,
        transparent_pass_batches,
        transparent_class_stats,
        depth_prepass_batches,
        depth_prepass_instances,
        gpu_instances_emitted,
        submitted_pipeline_pass_total,
        nontransparent_bins: arrangement.nontransparent_bins,
        nontransparent_binned_draws: arrangement.nontransparent_binned_draws,
        strict_sorted_draws: arrangement.strict_sorted_draws,
    }
}

/// Counts transparent draw classes in sorted draw order.
fn transparent_class_stats_from_sorted(
    draws: &[WorldMeshDrawItem],
) -> WorldMeshTransparentClassStats {
    let mut stats = WorldMeshTransparentClassStats::default();
    for draw in draws {
        stats.add(draw.batch_key.transparent_class);
    }
    stats
}

fn depth_prepass_counts(depth_groups: &[DrawGroup]) -> (usize, usize) {
    let mut batches = 0usize;
    let mut instances = 0usize;
    for group in depth_groups {
        batches += 1;
        instances += (group.instance_range.end - group.instance_range.start) as usize;
    }
    (batches, instances)
}

/// Captures draw-state diagnostics from the sorted draw list submitted by the forward pass.
pub fn state_rows_from_sorted(draws: &[WorldMeshDrawItem]) -> Vec<WorldMeshDrawStateRow> {
    draws
        .iter()
        .enumerate()
        .map(|(draw_index, item)| {
            let state = item.batch_key.render_state;
            WorldMeshDrawStateRow {
                draw_index,
                space_id: item.space_id.0,
                node_id: item.node_id,
                mesh_asset_id: item.mesh_asset_id,
                slot_index: item.slot_index,
                shader_asset_id: item.batch_key.shader_asset_id,
                material_asset_id: item.batch_key.material_asset_id,
                property_block_slot0: item.batch_key.property_block_slot0,
                pipeline: item.batch_key.pipeline.clone(),
                blend_mode: item.batch_key.blend_mode,
                is_overlay: item.is_overlay,
                sorting_order: item.sorting_order,
                collect_order: item.collect_order,
                skinned: item.skinned,
                alpha_blended: item.batch_key.alpha_blended,
                transparent_class: item.batch_key.transparent_class,
                requires_intersection_pass: item.batch_key.embedded_requires_intersection_pass,
                depth_write: state.depth_write,
                depth_compare: state.depth_compare,
                depth_offset: state
                    .depth_offset
                    .map(|offset| (offset.factor_bits(), offset.units())),
                stencil_enabled: state.stencil.enabled,
                stencil_reference: state.stencil.reference,
                stencil_compare: state.stencil.compare,
                stencil_pass_op: state.stencil.pass_op,
                stencil_read_mask: state.stencil.read_mask,
                stencil_write_mask: state.stencil.write_mask,
                color_mask: state.color_mask,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::materials::{MaterialBlendMode, MaterialDepthOffsetState};
    use crate::world_mesh::TransparentMaterialClass;
    use crate::world_mesh::test_fixtures::{DummyDrawItemSpec, dummy_world_mesh_draw_item};

    #[test]
    fn world_mesh_draw_stats_empty() {
        let s = stats_from_sorted(
            &[],
            None,
            WorldMeshDrawArrangementStats::default(),
            true,
            ShaderPermutation(0),
        );
        assert_eq!(s.batch_total, 0);
        assert_eq!(s.draws_total, 0);
        assert_eq!(s.instance_batch_total, 0);
        assert_eq!(s.intersect_pass_batches, 0);
        assert_eq!(s.transparent_pass_batches, 0);
        assert_eq!(s.transparent_class_stats, Default::default());
        assert_eq!(s.depth_prepass_batches, 0);
        assert_eq!(s.depth_prepass_instances, 0);
        assert_eq!(s.gpu_instances_emitted, 0);
        assert_eq!(s.submitted_pipeline_pass_total, 0);
    }

    #[test]
    fn world_mesh_draw_stats_single_batch() {
        let a = dummy_world_mesh_draw_item(DummyDrawItemSpec {
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
        let b = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 1,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id: 0,
            slot_index: 1,
            collect_order: 1,
            alpha_blended: false,
        });
        let draws = vec![a, b];
        let s = stats_from_sorted(
            &draws,
            None,
            WorldMeshDrawArrangementStats::default(),
            true,
            ShaderPermutation(0),
        );
        assert_eq!(s.batch_total, 1);
        assert_eq!(s.draws_total, 2);
        assert_eq!(s.rigid_draws, 2);
        assert_eq!(s.instance_batch_total, 1);
        assert_eq!(s.intersect_pass_batches, 0);
        assert_eq!(s.transparent_pass_batches, 0);
        assert_eq!(s.depth_prepass_batches, 1);
        assert_eq!(s.depth_prepass_instances, 2);
        assert_eq!(s.gpu_instances_emitted, 2);
        assert_eq!(s.submitted_pipeline_pass_total, 1);
    }

    #[test]
    fn world_mesh_draw_stats_counts_grab_pass_batches() {
        let mut draw = dummy_world_mesh_draw_item(DummyDrawItemSpec {
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
        draw.batch_key.embedded_uses_scene_color_snapshot = true;
        draw.batch_key.transparent_class = TransparentMaterialClass::GrabPassFilter;
        let s = stats_from_sorted(
            &[draw],
            None,
            WorldMeshDrawArrangementStats::default(),
            true,
            ShaderPermutation(0),
        );
        assert_eq!(s.instance_batch_total, 1);
        assert_eq!(s.intersect_pass_batches, 0);
        assert_eq!(s.transparent_pass_batches, 1);
        assert_eq!(s.transparent_class_stats.grab_pass_filter_draws, 1);
        assert_eq!(s.depth_prepass_batches, 0);
        assert_eq!(s.depth_prepass_instances, 0);
        assert_eq!(s.gpu_instances_emitted, 1);
    }

    #[test]
    fn world_mesh_draw_state_rows_capture_material_state() {
        let mut draw = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 7,
            property_block: Some(70),
            skinned: false,
            sorting_order: 3,
            mesh_asset_id: 4,
            node_id: 5,
            slot_index: 6,
            collect_order: 8,
            alpha_blended: true,
        });
        draw.batch_key.blend_mode = MaterialBlendMode::UnityBlend { src: 1, dst: 10 };
        draw.batch_key.render_state.depth_write = Some(false);
        draw.batch_key.render_state.depth_compare =
            Some(MaterialDepthCompareOverride::HostValue(8));
        draw.batch_key.render_state.depth_offset = MaterialDepthOffsetState::new(-1.0, -2);
        draw.batch_key.render_state.color_mask = Some(0);
        draw.batch_key.render_state.stencil.enabled = true;
        draw.batch_key.render_state.stencil.reference = 2;
        draw.batch_key.render_state.stencil.compare = 8;
        draw.batch_key.render_state.stencil.pass_op = 2;

        let rows = state_rows_from_sorted(&[draw]);
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.draw_index, 0);
        assert_eq!(row.material_asset_id, 7);
        assert_eq!(row.property_block_slot0, Some(70));
        assert_eq!(row.depth_write, Some(false));
        assert_eq!(
            row.depth_compare,
            Some(MaterialDepthCompareOverride::HostValue(8))
        );
        assert_eq!(row.depth_offset, Some(((-1.0f32).to_bits(), -2)));
        assert_eq!(row.color_mask, Some(0));
        assert!(row.stencil_enabled);
        assert_eq!(row.stencil_reference, 2);
        assert_eq!(
            row.blend_mode,
            MaterialBlendMode::UnityBlend { src: 1, dst: 10 }
        );
        assert_eq!(
            row.transparent_class,
            TransparentMaterialClass::OrderedAlpha
        );
    }
}
