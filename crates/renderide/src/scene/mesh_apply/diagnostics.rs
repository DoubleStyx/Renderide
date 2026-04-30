//! One-shot warn-once helpers shared by the static and skinned mesh apply paths.
//!
//! Both paths can encounter host messages whose `renderable_index` is out of range for the
//! current dense table (a host-renderer protocol drift symptom). The first occurrence per
//! `scene_id` is logged at `warn` level so the drift is diagnosable without flooding logs.

use std::collections::HashSet;
use std::sync::LazyLock;

use parking_lot::Mutex;

/// Once-per-scene dedup for [`super::static_meshes::apply_mesh_renderables_update_extracted`]
/// out-of-range `renderable_index` warnings.
pub(super) static STATIC_MESH_OOB_WARNED_SCENES: LazyLock<Mutex<HashSet<i32>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));

/// Once-per-scene dedup for skinned mesh state row out-of-range `renderable_index` warnings.
pub(super) static SKINNED_MESH_OOB_WARNED_SCENES: LazyLock<Mutex<HashSet<i32>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));

/// Once-per-scene dedup for skinned bone-index empty-buffer warnings.
pub(super) static BONE_INDEX_EMPTY_WARNED_SCENES: LazyLock<Mutex<HashSet<i32>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));

/// Logs the first out-of-range mesh state row per `scene_id`, then suppresses subsequent rows.
///
/// `kind` distinguishes the static vs. skinned mesh paths in the log line. The host's state-row
/// `renderable_index` should always be in `[0, len)` when the row is applied; an out-of-range
/// row indicates an addition row was dropped or a removals batch was skipped — silently
/// ignoring leaves the renderable invisible until the host re-emits.
pub(super) fn warn_oob_renderable_index_once(
    scene_id: i32,
    kind: &'static str,
    bad_index: usize,
    len: usize,
    warned: &Mutex<HashSet<i32>>,
) {
    let mut w = warned.lock();
    if w.insert(scene_id) {
        logger::warn!(
            "{kind} mesh state: renderable_index {bad_index} out of range (len={len}) in scene_id={scene_id}; row dropped silently. Suggests host-renderer protocol drift; subsequent occurrences in this scene are suppressed."
        );
    }
}
