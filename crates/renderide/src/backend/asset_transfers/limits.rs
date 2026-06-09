//! Admission limits for host-controlled asset transfer work.

/// Maximum single shared-memory upload payload admitted for renderer-owned staging.
pub(crate) const MAX_UPLOAD_PAYLOAD_BYTES: i32 = 256 * 1024 * 1024;

/// Maximum queued integration tasks retained under Host backpressure.
pub(crate) const MAX_ASSET_INTEGRATION_QUEUE_TASKS: usize = 4096;

/// Maximum active GStreamer-backed video players.
pub(crate) const MAX_ACTIVE_VIDEO_PLAYERS: usize = 64;

/// Maximum video load commands retained before GPU attachment.
pub(crate) const MAX_PENDING_VIDEO_TEXTURE_LOADS: usize = 64;

/// Returns whether a host descriptor length is small enough to copy or retain.
pub(crate) fn admit_descriptor_payload_len(kind: &str, asset_id: i32, len: i32) -> bool {
    if len <= 0 {
        return true;
    }
    if len > MAX_UPLOAD_PAYLOAD_BYTES {
        logger::warn!(
            "{kind} {asset_id}: rejected host payload length {} above cap {}",
            len,
            MAX_UPLOAD_PAYLOAD_BYTES
        );
        return false;
    }
    true
}
