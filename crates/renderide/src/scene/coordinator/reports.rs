//! Scene apply and render-world dirty reports.

use crate::shared::RenderingContext;

use super::super::ids::RenderSpaceId;
use super::super::overrides::MeshRendererOverrideTarget;

/// Static or skinned renderer table addressed by a render-world dirty event.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RenderWorldRendererKind {
    /// Dirty event targets [`crate::scene::render_space::RenderSpaceState::static_mesh_renderers`].
    Static,
    /// Dirty event targets [`crate::scene::render_space::RenderSpaceState::skinned_mesh_renderers`].
    Skinned,
}

/// PhotonDust render-buffer table addressed by a fine-grained prepared-snapshot dirty event.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RenderWorldParticleRendererKind {
    /// Billboard point-buffer renderer table.
    Billboard,
    /// Source-mesh particle renderer table.
    Mesh,
    /// Trail ribbon renderer table.
    Trail,
}

/// One generated particle renderer whose prepared rows need to be patched.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RenderWorldParticleRendererDirty {
    /// Host render space containing the renderer.
    pub space_id: RenderSpaceId,
    /// PhotonDust renderer table addressed by [`Self::renderable_index`].
    pub kind: RenderWorldParticleRendererKind,
    /// Dense renderer index in the selected table.
    pub renderable_index: usize,
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

/// One renderer row whose dynamic world bounds need to be refreshed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RenderWorldBoundsDirty {
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

/// Render context whose override targets need to be rescanned for a lightweight overlay.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderWorldContextOverrideDirty {
    /// Host render space containing the override rows.
    pub space_id: RenderSpaceId,
    /// Context whose prepared overlay may differ from the invariant base.
    pub context: RenderingContext,
}

/// Fine-grained dirty events consumed by backend render-world caches.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SceneRenderWorldDirtyReport {
    /// Render spaces that need a full retained-template refresh.
    pub full_spaces: Vec<RenderSpaceId>,
    /// Renderer rows that need retained-template refresh.
    pub renderers: Vec<RenderWorldRendererDirty>,
    /// Renderer rows whose live deformation inputs changed.
    ///
    /// Blendshape weights are consumed directly by the frame-global deform pass. The retained
    /// render world only needs to inspect these rows for a cached deformation-eligibility
    /// transition; ordinary active-to-active animation does not rebuild prepared rows. Bone-index
    /// changes are omitted entirely because the current prepared skin predicate receives
    /// `Some(slice)` regardless of the slice contents.
    pub deform_renderers: Vec<RenderWorldRendererDirty>,
    /// Renderer rows that only need dynamic world bounds refreshed.
    pub bounds: Vec<RenderWorldBoundsDirty>,
    /// Transform roots that need descendant renderer records refreshed after world-cache flush.
    pub transform_roots: Vec<RenderWorldTransformDirty>,
    /// Material override targets that need refresh only in matching render contexts.
    pub material_overrides: Vec<RenderWorldMaterialOverrideDirty>,
    /// Transform-override contexts whose exact overlay target set needs to be rescanned.
    pub context_overrides: Vec<RenderWorldContextOverrideDirty>,
    /// Particle render spaces whose generated-renderer membership changed.
    ///
    /// This invalidates only the particle suffix of a prepared space. Static and skinned retained
    /// templates remain reusable.
    pub particle_spaces: Vec<RenderSpaceId>,
    /// Individual particle renderer rows whose generated prepared runs need patching.
    pub particle_renderers: Vec<RenderWorldParticleRendererDirty>,
}

impl SceneRenderWorldDirtyReport {
    /// Returns whether the report contains no fine-grained render-world work.
    pub fn is_empty(&self) -> bool {
        self.full_spaces.is_empty()
            && self.renderers.is_empty()
            && self.deform_renderers.is_empty()
            && self.bounds.is_empty()
            && self.transform_roots.is_empty()
            && self.material_overrides.is_empty()
            && self.context_overrides.is_empty()
            && self.particle_spaces.is_empty()
            && self.particle_renderers.is_empty()
    }

    /// Records a renderer row that needs only dynamic bounds refresh.
    pub(super) fn note_bounds(
        &mut self,
        space_id: RenderSpaceId,
        kind: RenderWorldRendererKind,
        renderable_index: usize,
    ) {
        if self.full_spaces.contains(&space_id)
            || self.renderers.iter().any(|dirty| {
                dirty.space_id == space_id
                    && dirty.kind == kind
                    && dirty.renderable_index == renderable_index
            })
        {
            return;
        }
        let dirty = RenderWorldBoundsDirty {
            space_id,
            kind,
            renderable_index,
        };
        if !self.bounds.contains(&dirty) {
            self.bounds.push(dirty);
        }
    }

    /// Records a render space that needs a full retained-template refresh.
    pub(super) fn note_full_space(&mut self, id: RenderSpaceId) {
        if !self.full_spaces.contains(&id) {
            self.full_spaces.push(id);
        }
        self.renderers.retain(|dirty| dirty.space_id != id);
        self.deform_renderers.retain(|dirty| dirty.space_id != id);
        self.bounds.retain(|dirty| dirty.space_id != id);
    }

    /// Records a renderer row that needs retained-template refresh.
    pub(super) fn note_renderer(
        &mut self,
        space_id: RenderSpaceId,
        kind: RenderWorldRendererKind,
        renderable_index: usize,
    ) {
        if self.full_spaces.contains(&space_id) {
            return;
        }
        let dirty = RenderWorldRendererDirty {
            space_id,
            kind,
            renderable_index,
        };
        if !self.renderers.contains(&dirty) {
            self.renderers.push(dirty);
        }
        self.deform_renderers
            .retain(|candidate| *candidate != dirty);
        self.bounds.retain(|candidate| {
            candidate.space_id != space_id
                || candidate.kind != kind
                || candidate.renderable_index != renderable_index
        });
    }

    /// Records one renderer whose dynamic deformation inputs changed.
    pub(super) fn note_deform_renderer(
        &mut self,
        space_id: RenderSpaceId,
        kind: RenderWorldRendererKind,
        renderable_index: usize,
    ) {
        if self.full_spaces.contains(&space_id) {
            return;
        }
        let dirty = RenderWorldRendererDirty {
            space_id,
            kind,
            renderable_index,
        };
        if self.renderers.contains(&dirty) || self.deform_renderers.contains(&dirty) {
            return;
        }
        self.deform_renderers.push(dirty);
    }

    /// Records transform roots whose descendants may own cached renderer templates.
    pub(super) fn note_transform_roots<I>(&mut self, space_id: RenderSpaceId, roots: I)
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
    pub(super) fn note_material_override(
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

    /// Records a context-local override change without invalidating the invariant render world.
    pub(super) fn note_context_override(
        &mut self,
        space_id: RenderSpaceId,
        context: RenderingContext,
    ) {
        if !self
            .context_overrides
            .iter()
            .any(|dirty| dirty.space_id == space_id && dirty.context == context)
        {
            self.context_overrides
                .push(RenderWorldContextOverrideDirty { space_id, context });
        }
    }

    /// Records particle-renderer membership churn without invalidating retained mesh templates.
    pub(super) fn note_particle_space(&mut self, id: RenderSpaceId) {
        if !self.particle_spaces.contains(&id) {
            self.particle_spaces.push(id);
        }
        self.particle_renderers.retain(|dirty| dirty.space_id != id);
    }

    /// Records one generated particle renderer row unless its full particle suffix is dirty.
    pub(super) fn note_particle_renderer(
        &mut self,
        space_id: RenderSpaceId,
        kind: RenderWorldParticleRendererKind,
        renderable_index: usize,
    ) {
        if self.particle_spaces.contains(&space_id) {
            return;
        }
        let dirty = RenderWorldParticleRendererDirty {
            space_id,
            kind,
            renderable_index,
        };
        if !self.particle_renderers.contains(&dirty) {
            self.particle_renderers.push(dirty);
        }
    }
}

/// Scene changes observed while applying one host frame submission.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SceneApplyReport {
    /// Host frame index from [`crate::shared::FrameSubmitData::frame_index`].
    pub frame_index: i32,
    /// Render spaces present in the submission.
    pub submitted_spaces: Vec<RenderSpaceId>,
    /// Render spaces whose header or body payload may have changed scene-renderable state.
    pub changed_spaces: Vec<RenderSpaceId>,
    /// Render spaces whose extracted updates received complete fine-grained render-world
    /// classification, including the no-op case.
    ///
    /// Backends may use [`Self::changed_spaces`] as a conservative fallback only for spaces absent
    /// from this list. This prevents an exactly classified no-op or dynamic-deformation update from
    /// being promoted back into a full-space rebuild.
    pub render_world_classified_spaces: Vec<RenderSpaceId>,
    /// Render spaces whose reflection-probe sources or spatial placement may need refresh.
    pub reflection_probe_dirty_spaces: Vec<RenderSpaceId>,
    /// Render spaces removed because they were absent from the submission.
    pub removed_spaces: Vec<RenderSpaceId>,
    /// Fine-grained render-world dirty events for backend retained draw-template caches.
    pub render_world_dirty: SceneRenderWorldDirtyReport,
}

impl SceneApplyReport {
    /// Creates an empty report for `frame_index`.
    pub(super) fn new(frame_index: i32) -> Self {
        Self {
            frame_index,
            submitted_spaces: Vec::new(),
            changed_spaces: Vec::new(),
            render_world_classified_spaces: Vec::new(),
            reflection_probe_dirty_spaces: Vec::new(),
            removed_spaces: Vec::new(),
            render_world_dirty: SceneRenderWorldDirtyReport::default(),
        }
    }

    /// Records a render space that appeared in the host submission.
    pub(super) fn note_submitted_space(&mut self, id: RenderSpaceId) {
        self.submitted_spaces.push(id);
    }

    /// Records a render space that needs render-world maintenance.
    pub(super) fn note_changed_space(&mut self, id: RenderSpaceId) {
        if !self.changed_spaces.contains(&id) {
            self.changed_spaces.push(id);
        }
    }

    /// Marks one submitted space as completely classified for retained render-world effects.
    pub(super) fn note_render_world_classified_space(&mut self, id: RenderSpaceId) {
        if !self.render_world_classified_spaces.contains(&id) {
            self.render_world_classified_spaces.push(id);
        }
    }

    /// Records a render space whose reflection-probe selection state may need refresh.
    pub(super) fn note_reflection_probe_dirty_space(&mut self, id: RenderSpaceId) {
        if !self.reflection_probe_dirty_spaces.contains(&id) {
            self.reflection_probe_dirty_spaces.push(id);
        }
    }
}

/// World-cache flushes completed after scene apply.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SceneCacheFlushReport {
    /// Render spaces whose world transform caches were successfully refreshed.
    pub flushed_spaces: Vec<RenderSpaceId>,
}
