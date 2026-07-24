//! Shared-memory payload helpers for cooperative texture upload tasks. -xlinka

use std::sync::Arc;

use crate::assets::texture::TextureUploadError;
use crate::ipc::SharedMemoryAccessor;
use crate::shared::buffer::SharedMemoryBufferDescriptor;

const PAYLOAD_COPY_CHUNK_BYTES: usize = 1024 * 1024;

/// Owned bytes shared with background asset jobs. -xlinka
pub(super) type OwnedSharedMemoryPayload = Arc<Vec<u8>>;

/// Incremental shared-memory copy state. -xlinka
#[derive(Debug)]
pub(super) struct SharedMemoryPayloadCopy {
    bytes: Vec<u8>,
    expected_len: usize,
}

impl SharedMemoryPayloadCopy {
    pub(super) fn new(expected_len: usize) -> Result<Self, TextureUploadError> {
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(expected_len).map_err(|error| {
            TextureUploadError::from(format!(
                "failed to reserve {expected_len} payload bytes: {error}"
            ))
        })?;
        Ok(Self {
            bytes,
            expected_len,
        })
    }

    pub(super) fn copy_next_chunk(
        &mut self,
        shm: &mut SharedMemoryAccessor,
        descriptor: &SharedMemoryBufferDescriptor,
    ) -> Option<Result<Option<OwnedSharedMemoryPayload>, TextureUploadError>> {
        profiling::scope!("asset::shared_memory_payload_copy_chunk");
        shm.with_read_bytes(descriptor, |raw| {
            if raw.len() < self.expected_len {
                return Some(Err(TextureUploadError::from(format!(
                    "raw shorter than descriptor (need {}, got {})",
                    self.expected_len,
                    raw.len()
                ))));
            }
            let start = self.bytes.len();
            let end = start
                .saturating_add(PAYLOAD_COPY_CHUNK_BYTES)
                .min(self.expected_len);
            self.bytes.extend_from_slice(&raw[start..end]);
            if end < self.expected_len {
                return Some(Ok(None));
            }
            Some(Ok(Some(Arc::new(std::mem::take(&mut self.bytes)))))
        })
    }
}

/// Result of building an upload object while optionally preparing an owned payload copy. -xlinka
pub(super) struct SharedMemoryPayloadBuild<T> {
    /// Upload builder result produced from the shared-memory slice. -xlinka
    pub result: Result<T, TextureUploadError>,
    /// Cooperative descriptor copy retained for multi-step uploads. -xlinka
    pub payload_copy: Option<SharedMemoryPayloadCopy>,
}

/// Builds an uploader from a borrowed shared-memory slice and optionally owns the descriptor bytes.
///
/// Texture uploads can span multiple integration ticks. This helper keeps the shared-memory borrow
/// short while preserving the exact descriptor-window copy behavior used by the individual task
/// implementations. -xlinka
pub(super) fn build_with_optional_owned_payload<T>(
    shm: &mut SharedMemoryAccessor,
    descriptor: &SharedMemoryBufferDescriptor,
    build: impl FnOnce(&[u8]) -> Result<T, TextureUploadError>,
    needs_owned_payload: impl FnOnce(&T) -> bool,
) -> Option<SharedMemoryPayloadBuild<T>> {
    profiling::scope!("asset::shared_memory_payload_build");
    shm.with_read_bytes(descriptor, |raw| {
        let built = build(raw);
        let payload_copy = match built.as_ref() {
            Ok(value) if needs_owned_payload(value) => {
                let want = descriptor.length.max(0) as usize;
                if raw.len() < want {
                    return Some(SharedMemoryPayloadBuild {
                        result: Err(TextureUploadError::from(format!(
                            "raw shorter than descriptor (need {want}, got {})",
                            raw.len()
                        ))),
                        payload_copy: None,
                    });
                }
                let copy = match SharedMemoryPayloadCopy::new(want) {
                    Ok(copy) => copy,
                    Err(error) => {
                        return Some(SharedMemoryPayloadBuild {
                            result: Err(error),
                            payload_copy: None,
                        });
                    }
                };
                Some(copy)
            }
            _ => None,
        };
        Some(SharedMemoryPayloadBuild {
            result: built,
            payload_copy,
        })
    })
}
