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
    pub(super) fn note_full_space(&mut self, id: RenderSpaceId) {
        if !self.full_spaces.contains(&id) {
            self.full_spaces.push(id);
        }
    }

    /// Records a renderer row that needs retained-template refresh.
    pub(super) fn note_renderer(
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
}

/// World-cache flushes completed after scene apply.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SceneCacheFlushReport {
    /// Render spaces whose world transform caches were successfully refreshed.
    pub flushed_spaces: Vec<RenderSpaceId>,
}
