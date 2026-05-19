//! Owns all [`RenderSpaceState`](super::render_space::RenderSpaceState) instances and applies per-frame host data.

mod apply;
mod queries;

use hashbrown::HashMap;
use std::collections::HashSet;

use glam::Mat4;

use crate::assets::texture::{HostTextureAssetKind, pack_host_texture_id};
use crate::color_space::DEFAULT_SKYBOX_CLEAR_COLOR;
use crate::ipc::SharedMemoryAccessor;
use crate::shared::{
    BlitToDisplayState, FrameSubmitData, RenderSH2, RenderSpaceUpdate, RenderingContext,
};

use super::DrainedReflectionProbeRenderChanges;
use super::error::SceneError;
use super::ids::RenderSpaceId;
use super::lights::{
    LightCache, ResolvedLight, apply_light_renderables_update, apply_lights_buffer_renderers_update,
};
#[cfg(test)]
use super::math::multiply_root;
use super::overrides::{MeshRendererOverrideTarget, decode_packed_mesh_renderer_target};
use super::render_space::{RenderSpaceState, RenderSpaceView};
use super::transforms::TransformRemovalEvent;
use super::world::{WorldTransformCache, compute_world_matrices_for_space, ensure_cache_shapes};

use apply::{ExtractedRenderSpaceUpdate, extract_render_space_update, light_updates_view};

const PRIMARY_DESKTOP_DISPLAY_INDEX: i16 = 0;

/// Warns when more than one non-overlay render space is marked active (breaks main-camera assumptions).
fn warn_if_multiple_active_non_overlay_spaces(data: &FrameSubmitData) {
    let active_non_overlay = data
        .render_spaces
        .iter()
        .filter(|u| u.is_active && !u.is_overlay)
        .count();
    if active_non_overlay > 1 {
        logger::warn!(
            "FrameSubmitData: {active_non_overlay} active non-overlay render spaces (expected at most one for main camera parity)"
        );
    }
}

fn render_world_header_changed(
    space: Option<&RenderSpaceState>,
    update: &RenderSpaceUpdate,
) -> bool {
    let Some(space) = space else {
        return true;
    };
    space.is_active != update.is_active
        || space.is_overlay != update.is_overlay
        || space.view_position_is_external != update.view_position_is_external
}

fn extracted_update_affects_render_world(update: &ExtractedRenderSpaceUpdate) -> bool {
    update.transforms.is_some()
        || update.meshes.is_some()
        || update.skinned_meshes.is_some()
        || update.layers.is_some()
        || update.transform_overrides.is_some()
        || update.material_overrides.is_some()
}

fn note_render_world_dirty_for_extracted_update(
    report: &mut SceneApplyReport,
    space_id: RenderSpaceId,
    header_dirty: bool,
    current_node_count: usize,
    update: &ExtractedRenderSpaceUpdate,
) {
    if header_dirty {
        report.render_world_dirty.note_full_space(space_id);
    }
    if let Some(ref transforms) = update.transforms {
        note_transform_update_render_world_dirty(report, space_id, current_node_count, transforms);
    }
    if let Some(ref meshes) = update.meshes {
        note_mesh_update_render_world_dirty(
            report,
            space_id,
            RenderWorldRendererKind::Static,
            &meshes.removals,
            &meshes.additions,
            meshes
                .mesh_states
                .iter()
                .map(|state| state.renderable_index),
        );
    }
    if let Some(ref skinned_meshes) = update.skinned_meshes {
        note_skinned_mesh_update_render_world_dirty(report, space_id, skinned_meshes);
    }
    if update.layers.is_some() || update.transform_overrides.is_some() {
        report.render_world_dirty.note_full_space(space_id);
    }
    if let Some(ref material_overrides) = update.material_overrides {
        note_material_override_update_render_world_dirty(report, space_id, material_overrides);
    }
}

fn has_active_dense_indices(values: &[i32]) -> bool {
    values
        .iter()
        .take_while(|&&value| value >= 0)
        .next()
        .is_some()
}

fn note_mesh_update_render_world_dirty<I>(
    report: &mut SceneApplyReport,
    space_id: RenderSpaceId,
    kind: RenderWorldRendererKind,
    removals: &[i32],
    additions: &[i32],
    renderable_indices: I,
) where
    I: IntoIterator<Item = i32>,
{
    if has_active_dense_indices(removals) || has_active_dense_indices(additions) {
        report.render_world_dirty.note_full_space(space_id);
        return;
    }
    for renderable_index in renderable_indices {
        if renderable_index < 0 {
            break;
        }
        report
            .render_world_dirty
            .note_renderer(space_id, kind, renderable_index as usize);
    }
}

fn note_skinned_mesh_update_render_world_dirty(
    report: &mut SceneApplyReport,
    space_id: RenderSpaceId,
    skinned_meshes: &super::meshes::ExtractedSkinnedMeshRenderablesUpdate,
) {
    if has_active_dense_indices(&skinned_meshes.removals)
        || has_active_dense_indices(&skinned_meshes.additions)
    {
        report.render_world_dirty.note_full_space(space_id);
        return;
    }
    let kind = RenderWorldRendererKind::Skinned;
    for state in &skinned_meshes.mesh_states {
        if state.renderable_index < 0 {
            break;
        }
        report
            .render_world_dirty
            .note_renderer(space_id, kind, state.renderable_index as usize);
    }
    for assignment in &skinned_meshes.bone_assignments {
        if assignment.renderable_index < 0 {
            break;
        }
        report.render_world_dirty.note_renderer(
            space_id,
            kind,
            assignment.renderable_index as usize,
        );
    }
    for batch in &skinned_meshes.blendshape_update_batches {
        if batch.renderable_index < 0 {
            break;
        }
        report
            .render_world_dirty
            .note_renderer(space_id, kind, batch.renderable_index as usize);
    }
    for bounds in &skinned_meshes.bounds_updates {
        if bounds.renderable_index < 0 {
            break;
        }
        report
            .render_world_dirty
            .note_renderer(space_id, kind, bounds.renderable_index as usize);
    }
}

fn note_transform_update_render_world_dirty(
    report: &mut SceneApplyReport,
    space_id: RenderSpaceId,
    current_node_count: usize,
    transforms: &super::transforms::ExtractedTransformsUpdate,
) {
    if has_active_dense_indices(&transforms.removals)
        || (transforms.target_transform_count >= 0
            && transforms.target_transform_count as usize != current_node_count)
    {
        report.render_world_dirty.note_full_space(space_id);
        return;
    }
    let pose_roots = transforms
        .pose_updates
        .iter()
        .take_while(|pose| pose.transform_id >= 0)
        .map(|pose| pose.transform_id);
    let parent_roots = transforms
        .parent_updates
        .iter()
        .take_while(|parent| parent.transform_id >= 0)
        .map(|parent| parent.transform_id);
    report
        .render_world_dirty
        .note_transform_roots(space_id, pose_roots.chain(parent_roots));
}

fn note_material_override_update_render_world_dirty(
    report: &mut SceneApplyReport,
    space_id: RenderSpaceId,
    material_overrides: &super::overrides::ExtractedRenderMaterialOverridesUpdate,
) {
    if has_active_dense_indices(&material_overrides.removals)
        || has_active_dense_indices(&material_overrides.additions)
    {
        report.render_world_dirty.note_full_space(space_id);
        return;
    }
    for state in &material_overrides.states {
        if state.renderable_index < 0 {
            break;
        }
        let target = decode_packed_mesh_renderer_target(state.packed_mesh_renderer_index);
        match target {
            MeshRendererOverrideTarget::Static(_) | MeshRendererOverrideTarget::Skinned(_) => {
                report
                    .render_world_dirty
                    .note_material_override(space_id, state.context, target);
            }
            MeshRendererOverrideTarget::Unknown => {
                report.render_world_dirty.note_full_space(space_id);
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests;

/// Scene registry: one entry per host render space.
pub struct SceneCoordinator {
    /// Backed by [`hashbrown::HashMap`] for O(1) per-id lookup on the per-frame
    /// `apply_frame_submit` lift/reinsert path. Iteration order is non-deterministic; callers
    /// that need a stable order go through [`Self::render_space_ids`] which sorts ids by host
    /// `RenderSpaceId` value at iteration time.
    spaces: HashMap<RenderSpaceId, RenderSpaceState>,
    world_caches: HashMap<RenderSpaceId, WorldTransformCache>,
    world_dirty: HashSet<RenderSpaceId>,
    light_cache: LightCache,
    /// Reused in [`Self::flush_world_caches`] to avoid per-flush `Vec` allocation.
    world_dirty_flush_scratch: Vec<RenderSpaceId>,
    /// Reused in [`Self::remove_render_spaces_not_in_submit`].
    remove_spaces_scratch: Vec<RenderSpaceId>,
    /// Per-space transform swap-remove events emitted during Phase B of the current frame's
    /// apply. Consumed by Phase C so [`LightCache::fixup_for_transform_removals`] can roll
    /// cached `transform_id`s forward before the light update applies. Cleared at the top of
    /// every [`Self::apply_frame_submit`] so stale events never leak into later frames; the
    /// per-space [`Vec`] allocations are retained across frames to keep the steady-state path
    /// allocation-free.
    transform_removals_by_space: HashMap<RenderSpaceId, Vec<TransformRemovalEvent>>,
    /// Reused [`HashSet`] of render space ids seen in the current
    /// [`FrameSubmitData::render_spaces`]; cleared at the top of every
    /// [`Self::apply_frame_submit`] and consumed by
    /// [`Self::remove_render_spaces_not_in_submit`].
    apply_seen_scratch: HashSet<RenderSpaceId>,
    /// Reused per-space [`ExtractedRenderSpaceUpdate`] buffer for Phase A of every
    /// [`Self::apply_frame_submit`]; drained into Phase B, then refilled next frame.
    apply_extracted_scratch: Vec<ExtractedRenderSpaceUpdate>,
    /// Reused per-space work buffer for [`Self::apply_extracted_per_space`]'s Phase B drain.
    ///
    /// Holds one tuple per space whose state was lifted out of [`Self::spaces`] /
    /// [`Self::world_caches`] for the parallel apply. Drained in place after the loop so the
    /// allocation persists across frames; previously this was a fresh
    /// `Vec::with_capacity(extracted_per_space.len())` per frame.
    apply_work_scratch: Vec<ApplyWorkSlot>,
}

/// One per-space work slot held in [`SceneCoordinator::apply_work_scratch`].
struct ApplyWorkSlot {
    /// Render space identity for reinsert and dirty-cache tracking.
    id: RenderSpaceId,
    /// Lifted per-space scene state.
    space: RenderSpaceState,
    /// Lifted per-space world transform cache.
    cache: WorldTransformCache,
    /// Pre-extracted host update payload to apply.
    extracted: ExtractedRenderSpaceUpdate,
    /// Reused transform-removal side buffer for this work item.
    removal_events: Vec<TransformRemovalEvent>,
    /// Whether applying this slot dirtied the world transform cache.
    world_dirty: bool,
}

/// Static or skinned renderer table addressed by a render-world dirty event.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RenderWorldRendererKind {
    /// Dirty event targets [`RenderSpaceState::static_mesh_renderers`].
    Static,
    /// Dirty event targets [`RenderSpaceState::skinned_mesh_renderers`].
    Skinned,
}

/// One renderer row whose retained draw templates need to be refreshed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RenderWorldRendererDirty {
    /// Host render space containing the renderer.
    pub space_id: RenderSpaceId,
    /// Renderer table addressed by [`Self::renderable_index`].
    pub kind: RenderWorldRendererKind,
    /// Dense renderer index in the selected table.
    pub renderable_index: usize,
}

/// Transform roots whose descendant renderers need cached world-dependent template refresh.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RenderWorldTransformDirty {
    /// Host render space containing the transform roots.
    pub space_id: RenderSpaceId,
    /// Dense transform ids whose descendants may have changed world matrices.
    pub root_node_ids: Vec<i32>,
}

/// Material override row whose render-context-specific target needs template refresh.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderWorldMaterialOverrideDirty {
    /// Host render space containing the override row.
    pub space_id: RenderSpaceId,
    /// Render context the override applies to.
    pub context: RenderingContext,
    /// Static or skinned mesh renderer targeted by the override.
    pub target: MeshRendererOverrideTarget,
}

/// Fine-grained dirty events consumed by backend render-world caches.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SceneRenderWorldDirtyReport {
    /// Render spaces that need a full retained-template refresh.
    pub full_spaces: Vec<RenderSpaceId>,
    /// Renderer rows that need retained-template refresh.
    pub renderers: Vec<RenderWorldRendererDirty>,
    /// Transform roots that need descendant renderer records refreshed after world-cache flush.
    pub transform_roots: Vec<RenderWorldTransformDirty>,
    /// Material override targets that need refresh only in matching render contexts.
    pub material_overrides: Vec<RenderWorldMaterialOverrideDirty>,
}

impl SceneRenderWorldDirtyReport {
    /// Returns whether the report contains no fine-grained render-world work.
    pub fn is_empty(&self) -> bool {
        self.full_spaces.is_empty()
            && self.renderers.is_empty()
            && self.transform_roots.is_empty()
            && self.material_overrides.is_empty()
    }

    /// Records a render space that needs a full retained-template refresh.
    fn note_full_space(&mut self, id: RenderSpaceId) {
        if !self.full_spaces.contains(&id) {
            self.full_spaces.push(id);
        }
    }

    /// Records a renderer row that needs retained-template refresh.
    fn note_renderer(
        &mut self,
        space_id: RenderSpaceId,
        kind: RenderWorldRendererKind,
        renderable_index: usize,
    ) {
        self.renderers.push(RenderWorldRendererDirty {
            space_id,
            kind,
            renderable_index,
        });
    }

    /// Records transform roots whose descendants may own cached renderer templates.
    fn note_transform_roots<I>(&mut self, space_id: RenderSpaceId, roots: I)
    where
        I: IntoIterator<Item = i32>,
    {
        let mut root_node_ids = Vec::new();
        for root in roots {
            if root >= 0 && !root_node_ids.contains(&root) {
                root_node_ids.push(root);
            }
        }
        if !root_node_ids.is_empty() {
            self.transform_roots.push(RenderWorldTransformDirty {
                space_id,
                root_node_ids,
            });
        }
    }

    /// Records a context-specific material override target.
    fn note_material_override(
        &mut self,
        space_id: RenderSpaceId,
        context: RenderingContext,
        target: MeshRendererOverrideTarget,
    ) {
        self.material_overrides
            .push(RenderWorldMaterialOverrideDirty {
                space_id,
                context,
                target,
            });
    }
}

/// Scene changes observed while applying one host frame submission.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SceneApplyReport {
    /// Host frame index from [`FrameSubmitData::frame_index`].
    pub frame_index: i32,
    /// Render spaces present in the submission.
    pub submitted_spaces: Vec<RenderSpaceId>,
    /// Render spaces whose header or body payload may have changed scene-renderable state.
    pub changed_spaces: Vec<RenderSpaceId>,
    /// Render spaces removed because they were absent from the submission.
    pub removed_spaces: Vec<RenderSpaceId>,
    /// Fine-grained render-world dirty events for backend retained draw-template caches.
    pub render_world_dirty: SceneRenderWorldDirtyReport,
}

impl SceneApplyReport {
    /// Creates an empty report for `frame_index`.
    fn new(frame_index: i32) -> Self {
        Self {
            frame_index,
            submitted_spaces: Vec::new(),
            changed_spaces: Vec::new(),
            removed_spaces: Vec::new(),
            render_world_dirty: SceneRenderWorldDirtyReport::default(),
        }
    }

    /// Records a render space that appeared in the host submission.
    fn note_submitted_space(&mut self, id: RenderSpaceId) {
        self.submitted_spaces.push(id);
    }

    /// Records a render space that needs render-world maintenance.
    fn note_changed_space(&mut self, id: RenderSpaceId) {
        if !self.changed_spaces.contains(&id) {
            self.changed_spaces.push(id);
        }
    }
}

/// World-cache flushes completed after scene apply.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SceneCacheFlushReport {
    /// Render spaces whose world transform caches were successfully refreshed.
    pub flushed_spaces: Vec<RenderSpaceId>,
}

impl SceneCacheFlushReport {}

impl Default for SceneCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

impl SceneCoordinator {
    /// Empty registry.
    pub fn new() -> Self {
        Self {
            spaces: HashMap::new(),
            world_caches: HashMap::new(),
            world_dirty: HashSet::new(),
            light_cache: LightCache::new(),
            world_dirty_flush_scratch: Vec::new(),
            remove_spaces_scratch: Vec::new(),
            transform_removals_by_space: HashMap::new(),
            apply_seen_scratch: HashSet::new(),
            apply_extracted_scratch: Vec::new(),
            apply_work_scratch: Vec::new(),
        }
    }

    /// Mutable light cache ([`LightsBufferRendererSubmission`](crate::shared::LightsBufferRendererSubmission) store, tests).
    pub fn light_cache_mut(&mut self) -> &mut LightCache {
        &mut self.light_cache
    }

    /// Render space ids currently present, ordered by host id for deterministic traversal.
    ///
    /// The backing [`hashbrown::HashMap`] iterates in unspecified order, so this method copies the
    /// keys into a small owned [`Vec`] and sorts by [`RenderSpaceId`] before returning the
    /// iterator. The allocation is negligible at typical scene sizes (1-10 active render spaces);
    /// callers that have observed a sorted contract from the prior `BTreeMap` backing are
    /// preserved.
    pub fn render_space_ids(&self) -> impl Iterator<Item = RenderSpaceId> {
        let mut ids: Vec<RenderSpaceId> = self.spaces.keys().copied().collect();
        ids.sort_unstable_by_key(|id| id.0);
        ids.into_iter()
    }

    /// Number of host render spaces currently tracked.
    pub fn render_space_count(&self) -> usize {
        self.spaces.len()
    }

    /// Total static and skinned mesh renderables across all spaces.
    pub fn total_mesh_renderable_count(&self) -> usize {
        self.spaces
            .values()
            .map(|s| s.static_mesh_renderers.len() + s.skinned_mesh_renderers.len())
            .sum()
    }

    /// Appends render-context-aware world-space lights for `id` into `out`.
    ///
    /// This uses the same transform basis as view draw collection so clustered-light culling and
    /// forward shading see lights in the same world space as the meshes for that view.
    pub fn resolve_lights_for_render_context_into(
        &self,
        id: RenderSpaceId,
        context: RenderingContext,
        head_output_transform: Mat4,
        out: &mut Vec<ResolvedLight>,
    ) {
        let sid = id.0;
        self.light_cache.resolve_lights_into(
            sid,
            |transform_idx| {
                self.world_matrix_for_render_context(
                    id,
                    transform_idx,
                    context,
                    head_output_transform,
                )
            },
            out,
        );
    }

    /// Read-only access for debugging / future systems.
    pub fn space(&self, id: RenderSpaceId) -> Option<RenderSpaceView<'_>> {
        self.spaces.get(&id).map(RenderSpaceView::new)
    }

    /// Main non-overlay render space, matching the host's single active main-space expectation.
    pub fn active_main_space(&self) -> Option<RenderSpaceView<'_>> {
        self.spaces
            .values()
            .filter(|s| s.is_active && !s.is_overlay)
            .min_by_key(|s| s.id.0)
            .map(RenderSpaceView::new)
    }

    /// Ambient SH2 from the active non-overlay render space.
    pub fn active_main_ambient_light(&self) -> RenderSH2 {
        self.active_main_space()
            .map(|s| s.ambient_light())
            .unwrap_or_default()
    }

    /// Drains host changed-probe render requests after the latest scene apply.
    pub fn take_reflection_probe_render_changes(&mut self) -> DrainedReflectionProbeRenderChanges {
        let mut out = DrainedReflectionProbeRenderChanges::default();
        for space in self.spaces.values_mut() {
            let mut drained = super::reflection_probe::drain_reflection_probe_render_changes(space);
            out.completed.append(&mut drained.completed);
            out.scene_captures.append(&mut drained.scene_captures);
        }
        out
    }

    /// Latest explicit [`BlitToDisplayState`] targeting `display_index` from any active render
    /// space, or [`None`] if no active blit covers that display.
    ///
    /// When multiple blits target the same display, traversal is stable: active render spaces are
    /// visited by ascending [`RenderSpaceId`] and dense renderables by ascending table index, with
    /// later matches winning. `is_overlay` spaces are included so per-user and mirror blits keep
    /// working in overlay worlds. Entries whose state has not yet been initialized by a `states`
    /// row are skipped.
    pub fn active_blit_for_display(&self, display_index: i16) -> Option<BlitToDisplayState> {
        let mut latest: Option<BlitToDisplayState> = None;
        for id in self.render_space_ids() {
            let Some(space) = self.spaces.get(&id) else {
                continue;
            };
            if !space.is_active {
                continue;
            }
            for entry in &space.blit_to_displays {
                if !entry.state_initialized {
                    continue;
                }
                if entry.state.display_index != display_index {
                    continue;
                }
                if entry.state.texture_id < 0 {
                    continue;
                }
                latest = Some(entry.state);
            }
        }
        latest
    }

    /// Desktop-window blit source for `display_index`.
    ///
    /// Explicit host `BlitToDisplay` renderables win. Display zero can fall back to the active
    /// dashboard render-texture camera so desktop mode has a presentable dashboard while the
    /// overlay-camera path is still represented through regular render-texture views.
    pub fn desktop_blit_for_display(&self, display_index: i16) -> Option<BlitToDisplayState> {
        if let Some(state) = self.active_blit_for_display(display_index) {
            return Some(state);
        }
        if display_index == PRIMARY_DESKTOP_DISPLAY_INDEX {
            return self.synthesize_dash_blit_for_desktop_window();
        }
        None
    }

    /// Builds a synthetic [`BlitToDisplayState`] pointing at the dashboard camera's render
    /// texture for desktop presentation.
    ///
    /// The camera must be enabled, live in an active overlay render space, target a non-negative
    /// render texture, and use selective rendering. When multiple candidates exist, the
    /// lowest-depth camera wins.
    fn synthesize_dash_blit_for_desktop_window(&self) -> Option<BlitToDisplayState> {
        use crate::camera::camera_state_enabled;
        use crate::shared::CameraProjection;

        let mut best: Option<&crate::shared::CameraState> = None;
        for space in self.spaces.values() {
            if !space.is_active || !space.is_overlay {
                continue;
            }
            for entry in &space.cameras {
                let s = &entry.state;
                if !camera_state_enabled(s.flags) {
                    continue;
                }
                if s.projection != CameraProjection::Orthographic {
                    continue;
                }
                if s.render_texture_asset_id < 0 {
                    continue;
                }
                if s.selective_render_count <= 0 {
                    continue;
                }
                if best.is_none_or(|b| s.depth < b.depth) {
                    best = Some(s);
                }
            }
        }
        let cam = best?;
        let packed_texture_id = pack_host_texture_id(
            cam.render_texture_asset_id,
            HostTextureAssetKind::RenderTexture,
        )?;

        Some(BlitToDisplayState {
            renderable_index: -1,
            texture_id: packed_texture_id,
            background_color: DEFAULT_SKYBOX_CLEAR_COLOR,
            display_index: PRIMARY_DESKTOP_DISPLAY_INDEX,
            flags: 0,
            _padding: [0; 1],
        })
    }

    /// Current head-output render context for the main view.
    pub fn active_main_render_context(&self) -> RenderingContext {
        self.active_main_space()
            .map_or(RenderingContext::UserView, |space| {
                space.main_render_context()
            })
    }

    /// Cached world matrix from the host transform hierarchy (parent chain only).
    ///
    /// This matches object/light/bone placement: [`RenderSpaceState::root_transform`] is **not**
    /// applied here--it drives the view basis via [`RenderSpaceState::view_transform`], not mesh
    /// model matrices.
    pub fn world_matrix(&self, id: RenderSpaceId, transform_index: usize) -> Option<Mat4> {
        self.world_caches
            .get(&id)?
            .world_matrices
            .get(transform_index)
            .copied()
    }

    /// Hierarchy world matrix left-multiplied by [`RenderSpaceState::root_transform`].
    ///
    /// Use only when a host contract explicitly requires this composite. Default rendering uses
    /// [`Self::world_matrix`].
    #[cfg(test)]
    pub fn world_matrix_including_space_root(
        &self,
        id: RenderSpaceId,
        transform_index: usize,
    ) -> Option<Mat4> {
        let space = self.spaces.get(&id)?;
        let local = self.world_matrix(id, transform_index)?;
        Some(multiply_root(local, &space.root_transform))
    }

    /// Material override for the given renderer + slot in the given render context.
    pub fn overridden_material_asset_id(
        &self,
        space_id: RenderSpaceId,
        context: RenderingContext,
        skinned: bool,
        renderable_index: usize,
        slot_index: usize,
    ) -> Option<i32> {
        let space = self.spaces.get(&space_id)?;
        let target = if skinned {
            MeshRendererOverrideTarget::Skinned(renderable_index as i32)
        } else {
            MeshRendererOverrideTarget::Static(renderable_index as i32)
        };
        space.overridden_material_asset_id(context, target, slot_index)
    }

    /// Recomputes cached world matrices for every dirty space (no-op if caches clean).
    ///
    /// The per-space solve is data-independent (each [`WorldTransformCache`] is keyed by a
    /// distinct [`RenderSpaceId`]), so we drain dirty caches into a `Vec`, run the incremental
    /// solve in parallel via rayon, and reinsert successful results afterwards. On error the
    /// offending space is left marked dirty so the next flush retries; the first error observed
    /// is surfaced as the function result.
    pub fn flush_world_caches(&mut self) -> Result<SceneCacheFlushReport, SceneError> {
        profiling::scope!("scene::flush_world_caches");
        use rayon::prelude::*;

        let mut report = SceneCacheFlushReport::default();
        self.world_dirty_flush_scratch.clear();
        self.world_dirty_flush_scratch
            .extend(self.world_dirty.iter().copied());

        // Drop caches for dirty spaces that no longer exist and drain caches for surviving
        // spaces into a work vec. This runs on the main thread because it mutates `self`.
        let mut work: Vec<(RenderSpaceId, WorldTransformCache)> =
            Vec::with_capacity(self.world_dirty_flush_scratch.len());
        for id in self.world_dirty_flush_scratch.iter().copied() {
            if !self.spaces.contains_key(&id) {
                self.world_caches.remove(&id);
                self.world_dirty.remove(&id);
                continue;
            }
            let cache = self.world_caches.remove(&id).unwrap_or_default();
            work.push((id, cache));
        }

        if work.is_empty() {
            return Ok(report);
        }

        // `&self.spaces` is a shared borrow across rayon workers; `BTreeMap::get` is `Sync` for
        // `Sync` keys and values. Each task owns its own cache.
        let spaces = &self.spaces;
        let compute_one = |(id, mut cache): (RenderSpaceId, WorldTransformCache)| {
            // Space removed between drain and dispatch -- preserve cache as-is so the reinsert
            // step below drops it via the `Ok` path (caller treats this as a no-op).
            let Some(space) = spaces.get(&id) else {
                return (id, Ok(cache));
            };
            let n = space.nodes.len();
            ensure_cache_shapes(&mut cache, n, false);
            let result = compute_world_matrices_for_space(
                id.0,
                &space.nodes,
                &space.node_parents,
                &mut cache,
            );
            (id, result.map(|()| cache))
        };
        let results: Vec<(RenderSpaceId, Result<WorldTransformCache, SceneError>)> =
            if work.len() == 1 {
                work.into_iter().map(compute_one).collect()
            } else {
                work.into_par_iter().map(compute_one).collect()
            };

        let mut first_err: Option<SceneError> = None;
        for (id, result) in results {
            match result {
                Ok(cache) => {
                    self.world_caches.insert(id, cache);
                    self.world_dirty.remove(&id);
                    report.flushed_spaces.push(id);
                }
                Err(e) => {
                    if first_err.is_none() {
                        first_err = Some(e);
                    }
                    // Leave `world_dirty` set so the next flush retries this space.
                }
            }
        }

        if let Some(e) = first_err {
            return Err(e);
        }
        Ok(report)
    }

    /// Applies [`FrameSubmitData`]: transforms, meshes, skinned meshes, and lights in host order.
    ///
    /// Two-phase pipeline:
    ///
    /// 1. **Phase A (serial):** [`extract_render_space_update`] reads every shared-memory buffer
    ///    referenced by each [`RenderSpaceUpdate`] into owned vectors. Header fields
    ///    ([`RenderSpaceState::apply_update_header`]) are also applied here while we still hold a
    ///    serial borrow on the spaces map.
    /// 2. **Phase B (parallel above one space):** per-space mutation runs over the drained
    ///    `(RenderSpaceState, WorldTransformCache, ExtractedRenderSpaceUpdate)` tuples. Each
    ///    tuple owns disjoint state, so rayon workers cannot race.
    /// 3. **Phase C (serial):** light updates target the shared
    ///    [`crate::scene::lights::LightCache`] and run after the parallel apply.
    pub fn apply_frame_submit(
        &mut self,
        shm: &mut SharedMemoryAccessor,
        data: &FrameSubmitData,
    ) -> Result<SceneApplyReport, SceneError> {
        profiling::scope!("scene::apply_frame_submit");
        warn_if_multiple_active_non_overlay_spaces(data);
        let mut report = SceneApplyReport::new(data.frame_index);

        // Clear last frame's per-space removal events; Phase B refills them, Phase C consumes.
        // Retain the per-space `Vec` allocations to keep the steady-state path allocation-free.
        for v in self.transform_removals_by_space.values_mut() {
            v.clear();
        }

        // Reuse the cross-frame scratch HashSet and Vec; both are cleared on entry and put back
        // before this method returns so steady-state apply does not allocate either container.
        let mut seen = std::mem::take(&mut self.apply_seen_scratch);
        seen.clear();
        let mut extracted_per_space = std::mem::take(&mut self.apply_extracted_scratch);
        extracted_per_space.clear();
        extracted_per_space.reserve(data.render_spaces.len());

        // Phase A: serial pre-extract + ensure entries + apply header fields.
        {
            profiling::scope!("scene::apply_frame_submit::extract");
            for update in &data.render_spaces {
                let id = RenderSpaceId(update.id);
                seen.insert(id);
                report.note_submitted_space(id);
                let header_dirty = render_world_header_changed(self.spaces.get(&id), update);
                let space = self.spaces.entry(id).or_insert_with(|| RenderSpaceState {
                    id,
                    ..Default::default()
                });
                space.id = id;
                space.apply_update_header(update);
                let current_node_count = space.nodes.len();
                self.world_caches.entry(id).or_default();

                let extracted = extract_render_space_update(shm, update, data.frame_index)?;
                if header_dirty || extracted_update_affects_render_world(&extracted) {
                    report.note_changed_space(id);
                }
                note_render_world_dirty_for_extracted_update(
                    &mut report,
                    id,
                    header_dirty,
                    current_node_count,
                    &extracted,
                );
                extracted_per_space.push(extracted);
            }
        }

        // Phase B: per-space apply (parallel for >1 space, serial otherwise). Drains
        // `extracted_per_space`; the outer Vec keeps its capacity for next frame.
        self.apply_extracted_per_space(&mut extracted_per_space)?;

        // Phase C: light updates (still serial: shared LightCache). Before applying each space's
        // update we roll pre-existing cached `transform_id`s forward through any transform
        // swap-removes that ran in Phase B -- mirrors the host's `RenderableIndex` reindexing so a
        // light whose transform was swap-moved into a freed slot keeps pointing at it.
        {
            profiling::scope!("scene::apply_frame_submit::lights");
            for update in &data.render_spaces {
                let view = light_updates_view(update);
                if let Some(removals) = self
                    .transform_removals_by_space
                    .get(&RenderSpaceId(view.space_id))
                {
                    self.light_cache
                        .fixup_for_transform_removals(view.space_id, removals);
                }
                if let Some(lu) = view.lights_update {
                    apply_light_renderables_update(&mut self.light_cache, shm, lu, view.space_id)?;
                }
                if let Some(lbu) = view.lights_buffer_renderers_update {
                    apply_lights_buffer_renderers_update(
                        &mut self.light_cache,
                        shm,
                        lbu,
                        view.space_id,
                    )?;
                }
            }
        }

        self.remove_render_spaces_not_in_submit(&seen, &mut report.removed_spaces);

        // Restore the scratch containers (capacities retained for next frame).
        seen.clear();
        self.apply_seen_scratch = seen;
        debug_assert!(extracted_per_space.is_empty());
        self.apply_extracted_scratch = extracted_per_space;
        Ok(report)
    }

    /// Drops render spaces that were absent from this submit's id set.
    fn remove_render_spaces_not_in_submit(
        &mut self,
        seen: &HashSet<RenderSpaceId>,
        removed: &mut Vec<RenderSpaceId>,
    ) {
        self.remove_spaces_scratch.clear();
        self.remove_spaces_scratch
            .extend(self.spaces.keys().copied().filter(|id| !seen.contains(id)));
        for id in self.remove_spaces_scratch.iter().copied() {
            removed.push(id);
            self.light_cache.remove_space(id.0);
            self.spaces.remove(&id);
            self.world_caches.remove(&id);
            self.world_dirty.remove(&id);
            self.transform_removals_by_space.remove(&id);
        }
    }
}
