//! View-local mesh-swap LOD selection for world mesh draw collection.

use glam::{Mat4, Vec3};
use hashbrown::HashMap;

use crate::camera::{overlay_camera_view_matrix, view_matrix_for_world_mesh_render_space};
use crate::scene::{
    LodEntry, LodRendererKind, LodRendererRef, MeshRendererInstanceId, RenderSpaceId,
    RenderSpaceView, SceneCoordinator, SceneTransformRead, SkinnedMeshRenderer, StaticMeshRenderer,
};
use crate::world_mesh::culling::{
    MeshCullTarget, WorldMeshCullInput, mesh_world_geometry_for_cull_with_head,
};

use super::super::bitset::DenseBitSet;
use super::super::prepared_renderables::{FramePreparedLodEntry, FramePreparedLodGroup};
use super::DrawCollectionInputs;

/// Conservative relative screen height used when bounds cross the camera plane.
const CAMERA_INTERSECTING_RELATIVE_HEIGHT: f32 = 1.0;
/// Minimum homogeneous `w` accepted for screen-height projection.
const CLIP_W_EPS: f32 = 1e-6;
/// Per-view LOD decision map.
#[derive(Clone, Debug, Default)]
pub(super) struct LodVisibility {
    /// Per-space LOD bitsets and instance-id ordinal lookups.
    spaces: HashMap<RenderSpaceId, LodVisibilitySpace>,
}

impl LodVisibility {
    /// Creates a visibility map with stable renderer ordinals for the queried spaces.
    fn for_spaces(ctx: &DrawCollectionInputs<'_>, space_ids: &[RenderSpaceId]) -> Self {
        let mut spaces = HashMap::with_capacity(space_ids.len());
        for &space_id in space_ids {
            let Some(space) = ctx.scene_assets.scene.space(space_id) else {
                continue;
            };
            if !space.is_active() {
                continue;
            }
            let mut instance_to_ordinal = HashMap::with_capacity(
                space
                    .static_mesh_renderers()
                    .len()
                    .saturating_add(space.skinned_mesh_renderers().len()),
            );
            for renderer in space.static_mesh_renderers() {
                let ordinal = instance_to_ordinal.len();
                instance_to_ordinal.insert(renderer.instance_id, ordinal);
            }
            for renderer in space.skinned_mesh_renderers() {
                let ordinal = instance_to_ordinal.len();
                instance_to_ordinal.insert(renderer.base.instance_id, ordinal);
            }
            let renderer_count = instance_to_ordinal.len();
            let mut grouped = DenseBitSet::default();
            let mut selected = DenseBitSet::default();
            grouped.clear_and_resize(renderer_count);
            selected.clear_and_resize(renderer_count);
            spaces.insert(
                space_id,
                LodVisibilitySpace {
                    instance_to_ordinal,
                    grouped,
                    selected,
                },
            );
        }
        Self { spaces }
    }

    /// Returns whether `instance_id` may emit draws in this view.
    #[inline]
    pub(super) fn renderer_visible(
        &self,
        space_id: RenderSpaceId,
        renderer_ordinal: usize,
    ) -> bool {
        let Some(space) = self.spaces.get(&space_id) else {
            return true;
        };
        !space.grouped.contains(renderer_ordinal) || space.selected.contains(renderer_ordinal)
    }

    /// Returns whether `instance_id` may emit draws when only scene state is available.
    #[inline]
    #[cfg(test)]
    pub(super) fn renderer_visible_by_instance(
        &self,
        space_id: RenderSpaceId,
        instance_id: MeshRendererInstanceId,
    ) -> bool {
        let Some(space) = self.spaces.get(&space_id) else {
            return true;
        };
        let Some(&ordinal) = space.instance_to_ordinal.get(&instance_id) else {
            return true;
        };
        !space.grouped.contains(ordinal) || space.selected.contains(ordinal)
    }

    /// Marks a renderer as owned by an LOD group.
    fn mark_grouped(&mut self, space_id: RenderSpaceId, instance_id: MeshRendererInstanceId) {
        let Some(space) = self.spaces.get_mut(&space_id) else {
            return;
        };
        if let Some(&ordinal) = space.instance_to_ordinal.get(&instance_id) {
            space.grouped.insert(ordinal);
        }
    }

    /// Marks a grouped renderer as selected by the active LOD row.
    fn mark_selected(&mut self, space_id: RenderSpaceId, instance_id: MeshRendererInstanceId) {
        let Some(space) = self.spaces.get_mut(&space_id) else {
            return;
        };
        if let Some(&ordinal) = space.instance_to_ordinal.get(&instance_id) {
            space.selected.insert(ordinal);
        }
    }

    /// Marks a stable renderer ordinal as owned by an LOD group.
    fn mark_grouped_ordinal(&mut self, space_id: RenderSpaceId, renderer_ordinal: usize) {
        let Some(space) = self.spaces.get_mut(&space_id) else {
            return;
        };
        space.grouped.insert(renderer_ordinal);
    }

    /// Marks a stable renderer ordinal as selected by the active LOD row.
    fn mark_selected_ordinal(&mut self, space_id: RenderSpaceId, renderer_ordinal: usize) {
        let Some(space) = self.spaces.get_mut(&space_id) else {
            return;
        };
        space.selected.insert(renderer_ordinal);
    }

    /// Number of grouped renderer bits set across all spaces.
    #[cfg(test)]
    fn grouped_count(&self) -> usize {
        self.spaces
            .values()
            .map(|space| space.grouped.count_ones())
            .sum()
    }

    /// Number of selected renderer bits set across all spaces.
    #[cfg(test)]
    fn selected_count(&self) -> usize {
        self.spaces
            .values()
            .map(|space| space.selected.count_ones())
            .sum()
    }
}

/// One render space's LOD visibility bitsets.
#[derive(Clone, Debug, Default)]
struct LodVisibilitySpace {
    /// Stable renderer identity to current dense ordinal.
    instance_to_ordinal: HashMap<MeshRendererInstanceId, usize>,
    /// All live renderers referenced by at least one LOD group.
    grouped: DenseBitSet,
    /// Renderers selected by their owning group's chosen LOD.
    selected: DenseBitSet,
}

/// One LOD group scheduled for view-local selection.
#[derive(Clone, Copy)]
struct LodGroupWork<'a> {
    /// Render space that owns the LOD group.
    space_id: RenderSpaceId,
    /// Borrowed render-space tables used to resolve renderer references.
    space: RenderSpaceView<'a>,
    /// LOD rows in priority order.
    lods: &'a [LodEntry],
}

/// Renderer resolved from a stable LOD membership row.
struct ResolvedLodRenderer<'a> {
    /// Stable renderer-local identity.
    pub instance_id: MeshRendererInstanceId,
    /// Static or skinned renderer.
    pub kind: LodRendererKind,
    /// Renderer node id.
    pub node_id: i32,
    /// Whether this renderer is in the overlay layer.
    pub is_overlay: bool,
    /// Resident mesh asset id.
    pub mesh_asset_id: i32,
    /// Skinned renderer payload when `kind` is [`LodRendererKind::Skinned`].
    pub skinned_renderer: Option<&'a SkinnedMeshRenderer>,
}

/// LOD entry after resolving stale renderer ids against current scene tables.
struct ResolvedLodEntry {
    /// Threshold copied from scene LOD state.
    pub screen_relative_transition_height: f32,
    /// Live renderer ids in this entry.
    pub renderers: Vec<MeshRendererInstanceId>,
}

/// Builds the view-local LOD visibility map used by prepared and scene-walk collection.
pub(super) fn build_lod_visibility(
    ctx: &DrawCollectionInputs<'_>,
    space_ids: &[RenderSpaceId],
) -> LodVisibility {
    profiling::scope!("mesh::lod_visibility");
    let Some(culling) = ctx.view.culling else {
        return LodVisibility::default();
    };
    if let Some(prepared) = ctx.caches.prepared {
        let group_work = collect_prepared_lod_group_work(prepared.lod_groups(), space_ids);
        if group_work.is_empty() {
            return LodVisibility::default();
        }
        let mut visibility = LodVisibility::for_spaces(ctx, space_ids);
        {
            profiling::scope!("mesh::lod_visibility::select_prepared_groups");
            for group in group_work {
                select_prepared_group_lod(ctx, culling, group, &mut visibility);
            }
        }
        return visibility;
    }
    let capacity = lod_renderer_ref_capacity(ctx, space_ids);
    if capacity == 0 {
        return LodVisibility::default();
    }
    let group_work = collect_lod_group_work(ctx, space_ids);
    if group_work.is_empty() {
        return LodVisibility::default();
    }
    let mut visibility = LodVisibility::for_spaces(ctx, space_ids);
    {
        profiling::scope!("mesh::lod_visibility::select_scene_groups");
        for work in group_work {
            select_group_lod(
                ctx,
                culling,
                work.space_id,
                work.space,
                work.lods,
                &mut visibility,
            );
        }
    }
    visibility
}

/// Collects prepared LOD groups relevant to the requested render spaces.
fn collect_prepared_lod_group_work<'a>(
    groups: &'a [FramePreparedLodGroup],
    space_ids: &[RenderSpaceId],
) -> Vec<&'a FramePreparedLodGroup> {
    let mut work = Vec::new();
    for &space_id in space_ids {
        work.extend(groups.iter().filter(|group| group.space_id == space_id));
    }
    work
}

/// Selects one pre-resolved LOD group.
fn select_prepared_group_lod(
    ctx: &DrawCollectionInputs<'_>,
    culling: &WorldMeshCullInput<'_>,
    group: &FramePreparedLodGroup,
    visibility: &mut LodVisibility,
) {
    if group.lods.is_empty() {
        return;
    }
    for lod in &group.lods {
        for renderer in &lod.renderers {
            visibility.mark_grouped_ordinal(group.space_id, renderer.renderer_ordinal);
        }
    }

    let view_bounds = if group.world_aabb.is_some() {
        group.world_aabb.map(|bounds| (group.any_overlay, bounds))
    } else {
        scene_lod_group_view_bounds(ctx, group)
    };
    let selected_index = match view_bounds {
        Some((any_overlay, (wmin, wmax))) => {
            relative_screen_height_for_group(ctx, culling, group.space_id, any_overlay, wmin, wmax)
                .and_then(|relative_height| {
                    select_prepared_lod_index(&group.lods, relative_height, ctx.view.mesh_lod_bias)
                })
        }
        None => first_non_empty_prepared_lod(&group.lods),
    };

    let Some(selected_index) = selected_index else {
        return;
    };
    for renderer in &group.lods[selected_index].renderers {
        visibility.mark_selected_ordinal(group.space_id, renderer.renderer_ordinal);
    }
}

/// Recomputes view-dependent bounds for a prepared LOD group when cached bounds are unavailable.
fn scene_lod_group_view_bounds(
    ctx: &DrawCollectionInputs<'_>,
    group: &FramePreparedLodGroup,
) -> Option<(bool, (Vec3, Vec3))> {
    let space = ctx.scene_assets.scene.space(group.space_id)?;
    let scene_group = space.lod_groups().get(group.scene_group_index)?;
    let mut world_aabb = None;
    let mut any_overlay = false;
    for lod in &scene_group.lods {
        for renderer_ref in &lod.renderers {
            let Some(renderer) = resolve_lod_renderer(ctx, group.space_id, space, renderer_ref)
            else {
                continue;
            };
            any_overlay |= renderer.is_overlay;
            if let Some(bounds) = world_aabb_for_lod_renderer(ctx, group.space_id, &renderer) {
                union_aabb(&mut world_aabb, bounds);
            }
        }
    }
    world_aabb.map(|bounds| (any_overlay, bounds))
}

/// Estimates the renderer-ref capacity needed by one LOD worker chunk.
#[cfg(test)]
fn lod_group_chunk_capacity(
    total_renderer_refs: usize,
    total_groups: usize,
    chunk_groups: usize,
) -> usize {
    if total_renderer_refs == 0 || total_groups == 0 || chunk_groups == 0 {
        return 0;
    }
    total_renderer_refs
        .div_ceil(total_groups)
        .saturating_mul(chunk_groups)
}

/// Collects active LOD groups into a dense work list for serial or parallel selection.
fn collect_lod_group_work<'a>(
    ctx: &DrawCollectionInputs<'a>,
    space_ids: &[RenderSpaceId],
) -> Vec<LodGroupWork<'a>> {
    let mut work = Vec::new();
    for &space_id in space_ids {
        let Some(space) = ctx.scene_assets.scene.space(space_id) else {
            continue;
        };
        if !space.is_active() || space.lod_groups().is_empty() {
            continue;
        }
        for group in space.lod_groups() {
            work.push(LodGroupWork {
                space_id,
                space,
                lods: group.lods.as_slice(),
            });
        }
    }
    work
}

/// Counts renderer refs present in active LOD groups for visibility capacity planning.
fn lod_renderer_ref_capacity(ctx: &DrawCollectionInputs<'_>, space_ids: &[RenderSpaceId]) -> usize {
    space_ids
        .iter()
        .filter_map(|&space_id| ctx.scene_assets.scene.space(space_id))
        .filter(|space| space.is_active())
        .flat_map(|space| space.lod_groups().iter())
        .flat_map(|group| group.lods.iter())
        .map(|lod| lod.renderers.len())
        .sum()
}

/// Resolves, bounds, and selects one LOD group.
fn select_group_lod(
    ctx: &DrawCollectionInputs<'_>,
    culling: &WorldMeshCullInput<'_>,
    space_id: RenderSpaceId,
    space: RenderSpaceView<'_>,
    lods: &[LodEntry],
    visibility: &mut LodVisibility,
) {
    if lods.is_empty() {
        return;
    }

    let mut resolved_lods = Vec::with_capacity(lods.len());
    let mut world_aabb: Option<(Vec3, Vec3)> = None;
    let mut any_overlay = false;

    for lod in lods {
        let mut resolved_renderers = Vec::with_capacity(lod.renderers.len());
        for renderer_ref in &lod.renderers {
            let Some(renderer) = resolve_lod_renderer(ctx, space_id, space, renderer_ref) else {
                continue;
            };
            visibility.mark_grouped(space_id, renderer.instance_id);
            any_overlay |= renderer.is_overlay;
            if let Some(bounds) = world_aabb_for_lod_renderer(ctx, space_id, &renderer) {
                union_aabb(&mut world_aabb, bounds);
            }
            resolved_renderers.push(renderer.instance_id);
        }
        resolved_lods.push(ResolvedLodEntry {
            screen_relative_transition_height: lod.screen_relative_transition_height,
            renderers: resolved_renderers,
        });
    }

    if resolved_lods.iter().all(|lod| lod.renderers.is_empty()) {
        return;
    }

    let selected_index = match world_aabb {
        Some((wmin, wmax)) => {
            relative_screen_height_for_group(ctx, culling, space_id, any_overlay, wmin, wmax)
                .and_then(|relative_height| {
                    select_lod_index(&resolved_lods, relative_height, ctx.view.mesh_lod_bias)
                })
        }
        None => first_non_empty_lod(&resolved_lods),
    };

    let Some(selected_index) = selected_index else {
        return;
    };
    for instance_id in &resolved_lods[selected_index].renderers {
        visibility.mark_selected(space_id, *instance_id);
    }
}

/// Resolves a scene renderer from a stable LOD renderer reference.
fn resolve_lod_renderer<'a>(
    ctx: &DrawCollectionInputs<'_>,
    space_id: RenderSpaceId,
    space: RenderSpaceView<'a>,
    renderer_ref: &LodRendererRef,
) -> Option<ResolvedLodRenderer<'a>> {
    match renderer_ref.kind {
        LodRendererKind::Static => {
            let renderer = static_renderer_for_lod_ref(space, renderer_ref)?;
            if !renderer.emits_visible_color_draws()
                || renderer.mesh_asset_id < 0
                || renderer.node_id < 0
                || ctx
                    .scene_assets
                    .mesh_pool
                    .get(renderer.mesh_asset_id)
                    .is_none()
            {
                return None;
            }
            Some(ResolvedLodRenderer {
                instance_id: renderer.instance_id,
                kind: LodRendererKind::Static,
                node_id: renderer.node_id,
                is_overlay: renderer_is_overlay(ctx.scene_assets.scene, space_id, renderer.node_id),
                mesh_asset_id: renderer.mesh_asset_id,
                skinned_renderer: None,
            })
        }
        LodRendererKind::Skinned => {
            let renderer = skinned_renderer_for_lod_ref(space, renderer_ref)?;
            let base = &renderer.base;
            if !base.emits_visible_color_draws()
                || base.mesh_asset_id < 0
                || base.node_id < 0
                || ctx.scene_assets.mesh_pool.get(base.mesh_asset_id).is_none()
            {
                return None;
            }
            Some(ResolvedLodRenderer {
                instance_id: base.instance_id,
                kind: LodRendererKind::Skinned,
                node_id: base.node_id,
                is_overlay: renderer_is_overlay(ctx.scene_assets.scene, space_id, base.node_id),
                mesh_asset_id: base.mesh_asset_id,
                skinned_renderer: Some(renderer),
            })
        }
    }
}

/// Finds a static renderer by hint first, then by stable instance id.
fn static_renderer_for_lod_ref<'a>(
    space: RenderSpaceView<'a>,
    renderer_ref: &LodRendererRef,
) -> Option<&'a StaticMeshRenderer> {
    space
        .static_mesh_renderers()
        .get(renderer_ref.renderable_index_hint)
        .filter(|renderer| renderer.instance_id == renderer_ref.instance_id)
        .or_else(|| {
            space
                .static_mesh_renderers()
                .iter()
                .find(|renderer| renderer.instance_id == renderer_ref.instance_id)
        })
}

/// Finds a skinned renderer by hint first, then by stable instance id.
fn skinned_renderer_for_lod_ref<'a>(
    space: RenderSpaceView<'a>,
    renderer_ref: &LodRendererRef,
) -> Option<&'a SkinnedMeshRenderer> {
    space
        .skinned_mesh_renderers()
        .get(renderer_ref.renderable_index_hint)
        .filter(|renderer| renderer.base.instance_id == renderer_ref.instance_id)
        .or_else(|| {
            space
                .skinned_mesh_renderers()
                .iter()
                .find(|renderer| renderer.base.instance_id == renderer_ref.instance_id)
        })
}

/// Returns whether a renderer node is on the overlay layer.
fn renderer_is_overlay(scene: &SceneCoordinator, space_id: RenderSpaceId, node_id: i32) -> bool {
    node_id >= 0
        && SceneTransformRead::transform_is_in_overlay_layer(scene, space_id, node_id as usize)
}

/// Computes a renderer's world-space AABB for LOD group bounds.
fn world_aabb_for_lod_renderer(
    ctx: &DrawCollectionInputs<'_>,
    space_id: RenderSpaceId,
    renderer: &ResolvedLodRenderer<'_>,
) -> Option<(Vec3, Vec3)> {
    let mesh = ctx.scene_assets.mesh_pool.get(renderer.mesh_asset_id)?;
    let target = MeshCullTarget {
        scene: ctx.scene_assets.scene,
        space_id,
        mesh,
        skinned: renderer.kind == LodRendererKind::Skinned,
        skinned_renderer: renderer.skinned_renderer,
        node_id: renderer.node_id,
    };
    mesh_world_geometry_for_cull_with_head(
        &target,
        ctx.view.head_output_transform,
        ctx.view.render_context,
    )
    .world_aabb
}

/// Expands `dst` to include `bounds`.
fn union_aabb(dst: &mut Option<(Vec3, Vec3)>, bounds: (Vec3, Vec3)) {
    match dst {
        Some((min, max)) => {
            *min = min.min(bounds.0);
            *max = max.max(bounds.1);
        }
        None => *dst = Some(bounds),
    }
}

/// Computes group relative screen height for the active view.
fn relative_screen_height_for_group(
    ctx: &DrawCollectionInputs<'_>,
    culling: &WorldMeshCullInput<'_>,
    space_id: RenderSpaceId,
    is_overlay: bool,
    wmin: Vec3,
    wmax: Vec3,
) -> Option<f32> {
    if is_overlay {
        let view_proj = culling.proj.overlay_proj * overlay_camera_view_matrix();
        return Some(projected_aabb_relative_height(view_proj, wmin, wmax));
    }
    let space = ctx.scene_assets.scene.space(space_id)?;
    let view = culling
        .host_camera
        .explicit_world_to_view()
        .unwrap_or_else(|| view_matrix_for_world_mesh_render_space(ctx.scene_assets.scene, space));
    let first = if let Some((left, right)) = culling.proj.vr_stereo {
        let left_height = projected_aabb_relative_height(left, wmin, wmax);
        let right_height = projected_aabb_relative_height(right, wmin, wmax);
        return Some(left_height.max(right_height));
    } else {
        culling.proj.world_proj * view
    };
    Some(projected_aabb_relative_height(first, wmin, wmax))
}

/// Projects a world AABB and returns Unity-style relative screen height.
fn projected_aabb_relative_height(view_proj: Mat4, wmin: Vec3, wmax: Vec3) -> f32 {
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for corner in aabb_corners(wmin, wmax) {
        let clip = view_proj * corner.extend(1.0);
        if !clip.is_finite() || clip.w <= CLIP_W_EPS {
            return CAMERA_INTERSECTING_RELATIVE_HEIGHT;
        }
        let y = clip.y / clip.w;
        if !y.is_finite() {
            return CAMERA_INTERSECTING_RELATIVE_HEIGHT;
        }
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    ((max_y - min_y) * 0.5).clamp(0.0, CAMERA_INTERSECTING_RELATIVE_HEIGHT)
}

/// Returns the eight corners of an AABB.
fn aabb_corners(wmin: Vec3, wmax: Vec3) -> [Vec3; 8] {
    [
        Vec3::new(wmin.x, wmin.y, wmin.z),
        Vec3::new(wmax.x, wmin.y, wmin.z),
        Vec3::new(wmin.x, wmax.y, wmin.z),
        Vec3::new(wmax.x, wmax.y, wmin.z),
        Vec3::new(wmin.x, wmin.y, wmax.z),
        Vec3::new(wmax.x, wmin.y, wmax.z),
        Vec3::new(wmin.x, wmax.y, wmax.z),
        Vec3::new(wmax.x, wmax.y, wmax.z),
    ]
}

/// Selects the first LOD whose biased relative height meets its threshold.
fn select_lod_index(
    lods: &[ResolvedLodEntry],
    relative_height: f32,
    mesh_lod_bias: f32,
) -> Option<usize> {
    let bias = if mesh_lod_bias.is_finite() && mesh_lod_bias > 0.0 {
        mesh_lod_bias
    } else {
        1.0
    };
    let effective_height = relative_height.max(0.0) * bias;
    lods.iter().position(|lod| {
        effective_height >= sanitized_transition_height(lod.screen_relative_transition_height)
    })
}

/// Returns the first LOD row that still has any live renderer.
fn first_non_empty_lod(lods: &[ResolvedLodEntry]) -> Option<usize> {
    lods.iter().position(|lod| !lod.renderers.is_empty())
}

/// Selects the first prepared LOD whose biased relative height meets its threshold.
fn select_prepared_lod_index(
    lods: &[FramePreparedLodEntry],
    relative_height: f32,
    mesh_lod_bias: f32,
) -> Option<usize> {
    let bias = if mesh_lod_bias.is_finite() && mesh_lod_bias > 0.0 {
        mesh_lod_bias
    } else {
        1.0
    };
    let effective_height = relative_height.max(0.0) * bias;
    lods.iter().position(|lod| {
        effective_height >= sanitized_transition_height(lod.screen_relative_transition_height)
    })
}

/// Returns the first prepared LOD row that still has any live renderer.
fn first_non_empty_prepared_lod(lods: &[FramePreparedLodEntry]) -> Option<usize> {
    lods.iter().position(|lod| !lod.renderers.is_empty())
}

/// Sanitizes host threshold values for robust selection.
fn sanitized_transition_height(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a resolved LOD entry with one renderer so selection can ignore emptiness.
    fn lod(threshold: f32, id: u64) -> ResolvedLodEntry {
        ResolvedLodEntry {
            screen_relative_transition_height: threshold,
            renderers: vec![MeshRendererInstanceId(id)],
        }
    }

    #[test]
    fn select_lod_index_uses_first_matching_threshold() {
        let lods = [lod(0.6, 1), lod(0.2, 2)];

        assert_eq!(select_lod_index(&lods, 0.35, 2.0), Some(0));
        assert_eq!(select_lod_index(&lods, 0.25, 1.0), Some(1));
    }

    #[test]
    fn select_lod_index_returns_none_below_final_threshold() {
        let lods = [lod(0.6, 1), lod(0.2, 2)];

        assert_eq!(select_lod_index(&lods, 0.1, 1.0), None);
    }

    #[test]
    fn first_non_empty_lod_falls_back_when_bounds_are_unavailable() {
        let lods = [
            ResolvedLodEntry {
                screen_relative_transition_height: 0.8,
                renderers: Vec::new(),
            },
            lod(0.2, 2),
        ];

        assert_eq!(first_non_empty_lod(&lods), Some(1));
    }

    #[test]
    fn projected_aabb_relative_height_measures_ndc_height() {
        let height = projected_aabb_relative_height(
            Mat4::IDENTITY,
            Vec3::new(-0.5, -0.25, 0.0),
            Vec3::new(0.5, 0.25, 0.0),
        );

        assert!((height - 0.25).abs() < 1e-6);
    }

    #[test]
    fn lod_visibility_keeps_ungrouped_renderers_visible() {
        let visibility = LodVisibility::default();

        assert!(
            visibility.renderer_visible_by_instance(RenderSpaceId(1), MeshRendererInstanceId(42))
        );
    }

    #[test]
    fn lod_visibility_hides_grouped_unselected_renderers() {
        let mut visibility = LodVisibility::default();
        let grouped = MeshRendererInstanceId(1);
        let selected = MeshRendererInstanceId(2);
        let space_id = RenderSpaceId(1);
        let mut instance_to_ordinal = HashMap::new();
        instance_to_ordinal.insert(grouped, 0);
        instance_to_ordinal.insert(selected, 1);
        let mut space = LodVisibilitySpace {
            instance_to_ordinal,
            grouped: DenseBitSet::default(),
            selected: DenseBitSet::default(),
        };
        space.grouped.insert(0);
        space.grouped.insert(1);
        space.selected.insert(1);
        visibility.spaces.insert(space_id, space);

        assert!(!visibility.renderer_visible_by_instance(space_id, grouped));
        assert!(visibility.renderer_visible_by_instance(space_id, selected));
        assert_eq!(visibility.grouped_count(), 2);
        assert_eq!(visibility.selected_count(), 1);
    }

    #[test]
    fn lod_group_chunk_capacity_scales_refs_by_chunk_size() {
        assert_eq!(lod_group_chunk_capacity(0, 8, 4), 0);
        assert_eq!(lod_group_chunk_capacity(17, 8, 4), 12);
        assert_eq!(lod_group_chunk_capacity(17, 8, 0), 0);
    }
}
