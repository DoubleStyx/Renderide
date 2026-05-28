//! Shared-memory writer helpers used by asset upload modules.

use std::path::Path;

use renderide_shared::{SharedMemoryWriter, SharedMemoryWriterConfig};

use crate::error::HarnessError;

/// Opens a session-scoped shared-memory writer and copies `bytes` into it at offset zero.
pub(super) fn open_writer(
    prefix: &str,
    backing_dir: &Path,
    buffer_id: i32,
    bytes: &[u8],
    label: &str,
) -> Result<SharedMemoryWriter, HarnessError> {
    let cfg = SharedMemoryWriterConfig {
        prefix: prefix.to_string(),
        destroy_on_drop: true,
        dir_override: Some(backing_dir.to_path_buf()),
    };
    let mut writer = SharedMemoryWriter::open(cfg, buffer_id, bytes.len()).map_err(|e| {
        HarnessError::QueueOptions(format!(
            "SharedMemoryWriter::open({label} buffer={buffer_id}, cap={}): {e}",
            bytes.len(),
        ))
    })?;
    writer
        .write_at(0, bytes)
        .map_err(|e| HarnessError::QueueOptions(format!("write {label} bytes: {e}")))?;
    writer.flush();
    Ok(writer)
}
