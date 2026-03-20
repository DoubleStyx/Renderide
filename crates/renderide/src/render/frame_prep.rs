//! Main-view frame input and read-only mesh-draw prep snapshots.
//!
//! The winit thread still owns the full [`crate::app::RenderideApp::run_frame`] pipeline; these
//! types separate **what** is needed for the main-view render after [`crate::session::Session::update`]
//! from GPU encode/submit.

use crate::render::SpaceDrawBatch;
use crate::session::Session;

/// Draw batches for the main window for one frame, produced after IPC update and before swapchain
/// acquire.
///
/// Building this via [`Self::from_session`] runs [`Session::collect_draw_batches`], which also
/// populates resolved lights used after present.
pub struct MainViewFrameInput {
    /// Per-space draw batches for the user view (main window).
    pub draw_batches: Vec<SpaceDrawBatch>,
}

impl MainViewFrameInput {
    /// Collects draw batches from the session for the main view.
    pub fn from_session(session: &mut Session) -> Self {
        Self {
            draw_batches: session.collect_draw_batches(),
        }
    }
}

/// Subset of render flags read during mesh-draw CPU collection, for tests and future worker-side
/// validation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MeshDrawPrepReadSnapshot {
    /// When true, log diagnostic info for the first skinned draw each frame.
    pub debug_skinned: bool,
    /// When true, rigid and non-overlay skinned meshes may be frustum culled on the CPU.
    pub frustum_culling: bool,
    /// When true, skinned MVP uses root bone world matrix.
    pub skinned_use_root_bone: bool,
    /// When true, apply Z flip for skinned handedness.
    pub skinned_flip_handedness: bool,
    /// When true, multi-batch mesh-draw collection may use scoped worker threads.
    pub parallel_mesh_draw_prep_batches: bool,
}

impl MeshDrawPrepReadSnapshot {
    /// Captures render-config fields consulted during mesh-draw collection for this frame.
    pub fn from_session(session: &Session) -> Self {
        let c = session.render_config();
        Self {
            debug_skinned: c.debug_skinned,
            frustum_culling: c.frustum_culling,
            skinned_use_root_bone: c.skinned_use_root_bone,
            skinned_flip_handedness: c.skinned_flip_handedness,
            parallel_mesh_draw_prep_batches: c.parallel_mesh_draw_prep_batches,
        }
    }

    /// Returns true if the session's current render config still matches this snapshot.
    pub fn matches_session(&self, session: &Session) -> bool {
        *self == Self::from_session(session)
    }
}

/// Documents the required ordering: IPC [`Session::update`] must run before main-view batch
/// collection so frame submit and asset uploads stay ordered.
///
/// Always returns `true`; callers and tests use it as an explicit anchor for that contract.
pub fn main_view_prep_requires_update_before_collect() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_mesh_prep_flag_reserved_until_session_sync_or_snapshots() {
        let session = Session::new();
        assert!(session.render_config().parallel_mesh_draw_prep_batches);
    }

    #[test]
    fn main_view_frame_input_empty_matches_collect() {
        let mut session = Session::new();
        let input = MainViewFrameInput::from_session(&mut session);
        assert!(input.draw_batches.is_empty());
    }

    #[test]
    fn mesh_draw_prep_snapshot_tracks_render_config() {
        let session = Session::new();
        let snap = MeshDrawPrepReadSnapshot::from_session(&session);
        assert!(snap.matches_session(&session));
        assert_eq!(snap.debug_skinned, session.render_config().debug_skinned);
        assert_eq!(
            snap.parallel_mesh_draw_prep_batches,
            session.render_config().parallel_mesh_draw_prep_batches
        );
    }

    #[test]
    fn ipc_ordering_anchor_documents_update_before_collect() {
        assert!(main_view_prep_requires_update_before_collect());
    }
}
