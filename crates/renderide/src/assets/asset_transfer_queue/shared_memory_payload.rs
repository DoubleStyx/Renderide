//! Shared-memory payload helpers for cooperative texture upload tasks.

use std::sync::Arc;

use crate::assets::texture::TextureUploadError;
use crate::ipc::SharedMemoryAccessor;
use crate::shared::buffer::SharedMemoryBufferDescriptor;

/// Result of building an upload object while optionally retaining an owned payload copy.
pub(super) struct SharedMemoryPayloadBuild<T> {
    /// Upload builder result produced from the shared-memory slice.
    pub result: Result<T, TextureUploadError>,
    /// Descriptor-window bytes retained for multi-step uploads.
    pub payload: Arc<[u8]>,
}

/// Builds an uploader from a borrowed shared-memory slice and optionally owns the descriptor bytes.
///
/// Texture uploads can span multiple integration ticks. This helper keeps the shared-memory borrow
/// short while preserving the exact descriptor-window copy behavior used by the individual task
/// implementations.
pub(super) fn build_with_optional_owned_payload<T>(
    shm: &mut SharedMemoryAccessor,
    descriptor: &SharedMemoryBufferDescriptor,
    build: impl FnOnce(&[u8]) -> Result<T, TextureUploadError>,
    needs_owned_payload: impl FnOnce(&T) -> bool,
) -> Option<SharedMemoryPayloadBuild<T>> {
    profiling::scope!("asset::shared_memory_payload_build");
    shm.with_read_bytes(descriptor, |raw| {
        let built = build(raw);
        let payload = match built.as_ref() {
            Ok(value) if needs_owned_payload(value) => {
                profiling::scope!("asset::shared_memory_payload_copy");
                let want = descriptor.length.max(0) as usize;
                if raw.len() < want {
                    return Some(SharedMemoryPayloadBuild {
                        result: Err(TextureUploadError::from(format!(
                            "raw shorter than descriptor (need {want}, got {})",
                            raw.len()
                        ))),
                        payload: Arc::<[u8]>::from([]),
                    });
                }
                Arc::from(&raw[..want])
            }
            _ => Arc::<[u8]>::from([]),
        };
        Some(SharedMemoryPayloadBuild {
            result: built,
            payload,
        })
    })
}
