//! Asset ingestion and GPU resource handles (skeleton).

pub mod mesh;
pub mod texture;

use std::collections::VecDeque;

/// Opaque host asset identifier (meshes, textures, shaders, etc.).
#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct AssetId(pub i32);

/// CPU-side placeholder for IPC payloads waiting for upload workers.
#[derive(Clone, Debug, Default)]
pub struct PendingUploadMeta {
    pub asset_id: AssetId,
    pub approx_bytes: u64,
}

/// Holds generational handles and a bounded queue of pending work (mirrors “integrator” staging).
#[derive(Debug, Default)]
pub struct AssetSubsystem {
    pending: VecDeque<PendingUploadMeta>,
}

impl AssetSubsystem {
    /// Records metadata for a future upload path (no GPU work yet).
    pub fn enqueue_pending(&mut self, meta: PendingUploadMeta) {
        self.pending.push_back(meta);
    }

    /// Returns queued upload metadata without performing transfers.
    pub fn drain_pending_meta(&mut self) -> Vec<PendingUploadMeta> {
        self.pending.drain(..).collect()
    }
}
