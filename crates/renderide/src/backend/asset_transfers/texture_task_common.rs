//! Shared helpers for cooperative texture-family upload tasks.

use std::sync::Arc;

use crate::gpu::{GpuQueueAccessGate, GpuQueueAccessMode};
use crate::ipc::DualQueueIpc;
use crate::shared::RendererCommand;

use super::integrator::StepResult;
use super::reliable_ack::send_background_reliable;

/// GPU handles and queue-access policy shared by one texture-family upload step.
#[derive(Clone, Copy)]
pub(super) struct TextureTaskGpu<'a> {
    /// Device used by texture decode and upload planning paths.
    pub(super) device: &'a Arc<wgpu::Device>,
    /// Queue used for texture writes.
    pub(super) queue: &'a wgpu::Queue,
    /// Queue access gate shared with submit and OpenXR queue calls.
    pub(super) queue_access_gate: &'a GpuQueueAccessGate,
    /// Queue-gate acquisition policy for texture writes.
    pub(super) queue_access_mode: GpuQueueAccessMode,
}

/// Returns a resident texture handle or logs a consistent missing-resource warning.
pub(super) fn resident_texture_arc(
    kind: &'static str,
    asset_id: i32,
    texture: Option<Arc<wgpu::Texture>>,
) -> Option<Arc<wgpu::Texture>> {
    texture.or_else(|| {
        logger::warn!("{kind} {asset_id}: missing GPU texture during integration step");
        None
    })
}

/// Logs a missing shared-memory payload and terminates the upload task.
pub(super) fn missing_payload(kind: &'static str, asset_id: i32) -> StepResult {
    logger::warn!("{kind} {asset_id}: shared memory slice missing");
    StepResult::Done
}

/// Logs an upload failure and terminates the upload task.
pub(super) fn failed_upload(
    kind: &'static str,
    asset_id: i32,
    error: &crate::assets::texture::TextureUploadError,
) -> StepResult {
    logger::warn!("{kind} {asset_id}: upload failed: {error}");
    StepResult::Done
}

/// Sends a background IPC result when the renderer is connected to a host.
pub(super) fn send_background_result(
    ipc: &mut Option<&mut DualQueueIpc>,
    command: RendererCommand,
) {
    let _ = send_background_reliable(ipc, command, || {
        "asset upload: failed to enqueue reliable background result".to_string()
    });
}

/// Returns whether an upload may write without mixing native storage orientations.
pub(super) fn storage_orientation_allows_upload(
    kind: &'static str,
    asset_id: i32,
    mip_levels_resident: u32,
    resident_storage_v_inverted: bool,
    upload_storage_v_inverted: bool,
    mismatch_detail: &'static str,
) -> bool {
    if mip_levels_resident > 0 && resident_storage_v_inverted != upload_storage_v_inverted {
        logger::warn!(
            "{kind} {asset_id}: upload storage orientation mismatch (resident inverted={}, upload inverted={}); aborting to avoid mixed-orientation {mismatch_detail}",
            resident_storage_v_inverted,
            upload_storage_v_inverted
        );
        return false;
    }
    true
}

/// Returns whether a post-write residency update may record the upload orientation.
pub(super) fn storage_orientation_allows_mark(
    kind: &'static str,
    asset_id: i32,
    mip_levels_resident: u32,
    resident_storage_v_inverted: bool,
    upload_storage_v_inverted: bool,
    phase: &'static str,
) -> bool {
    if mip_levels_resident > 0 && resident_storage_v_inverted != upload_storage_v_inverted {
        logger::warn!(
            "{kind} {asset_id}: upload storage orientation mismatch {phase} (resident inverted={}, upload inverted={})",
            resident_storage_v_inverted,
            upload_storage_v_inverted
        );
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_orientation_upload_allows_empty_resident_texture() {
        assert!(storage_orientation_allows_upload(
            "texture", 7, 0, false, true, "mips",
        ));
    }

    #[test]
    fn storage_orientation_upload_rejects_mixed_resident_texture() {
        assert!(!storage_orientation_allows_upload(
            "texture", 7, 1, false, true, "mips",
        ));
    }

    #[test]
    fn storage_orientation_mark_matches_upload_policy() {
        assert!(storage_orientation_allows_mark(
            "cubemap",
            3,
            4,
            true,
            true,
            "after write",
        ));
        assert!(!storage_orientation_allows_mark(
            "cubemap",
            3,
            4,
            true,
            false,
            "after write",
        ));
    }

    #[test]
    fn missing_payload_and_failed_upload_finish_task() {
        assert_eq!(missing_payload("texture", 1), StepResult::Done);
        assert_eq!(
            failed_upload(
                "texture",
                1,
                &crate::assets::texture::TextureUploadError::from("bad upload"),
            ),
            StepResult::Done,
        );
    }
}
