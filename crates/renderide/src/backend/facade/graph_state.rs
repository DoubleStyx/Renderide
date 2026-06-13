//! Render-graph lifetime state owned by [`super::RenderBackend`].
//!
//! This keeps graph cache/history/transient ownership together instead of scattering long-lived
//! graph resources across the backend facade.

use crate::backend::graph::MainGraphPostProcessingResources;
use crate::camera::ViewId;
use crate::frame_upload_batch::FrameUploadBatchStats;
use crate::gpu::driver_thread::SubmitToken;
use crate::render_graph::{GraphCache, TransientPool, context::GraphResolvedResources};
use crate::upload_arena::PersistentUploadArena;

use super::super::{HistoryRegistry, ViewResourceRegistry};

/// Transient graph resources waiting for their driver-thread submit token before pool release.
pub(super) struct PendingTransientRelease {
    /// Driver submit token that must be complete before these resources can re-enter the pool.
    pub(super) token: SubmitToken,
    /// Resolved graph resources retained until the submit reaches `Queue::submit`.
    pub(super) resources: Vec<GraphResolvedResources>,
}

/// Long-lived render-graph resources retained across frames.
pub(super) struct RenderGraphState {
    /// Cached compiled frame graph keyed by the shared render-graph cache inputs.
    pub(super) frame_graph_cache: GraphCache,
    /// Render-graph transient texture/buffer pool retained across frames.
    transient_pool: TransientPool,
    /// Persistent ping-pong resources used by graph history slots
    /// (`ImportSource::PingPong` / `BufferImportSource::PingPong`).
    history_registry: HistoryRegistry,
    /// Persistent staging-buffer arena for frame upload copies.
    upload_arena: PersistentUploadArena,
    /// Latest upload drain stats published by graph execution for diagnostics.
    latest_upload_stats: FrameUploadBatchStats,
    /// Transient resources recorded into queued submits but not yet reusable.
    pending_transient_releases: Vec<PendingTransientRelease>,
    /// Retained logical-view ownership for every backend cache that lives beyond one frame.
    view_resources: ViewResourceRegistry,
    /// Post-processing resources that must survive compiled graph rebuilds.
    post_processing_resources: MainGraphPostProcessingResources,
}

impl RenderGraphState {
    /// Creates empty graph state before GPU attach.
    pub(super) fn new() -> Self {
        Self {
            frame_graph_cache: GraphCache::default(),
            transient_pool: TransientPool::new(),
            history_registry: HistoryRegistry::new(),
            upload_arena: PersistentUploadArena::new(),
            latest_upload_stats: FrameUploadBatchStats::default(),
            pending_transient_releases: Vec::new(),
            view_resources: ViewResourceRegistry::new(),
            post_processing_resources: MainGraphPostProcessingResources::default(),
        }
    }

    /// Mutable graph transient pool.
    pub(super) fn transient_pool_mut(&mut self) -> &mut TransientPool {
        &mut self.transient_pool
    }

    /// Immutable graph transient pool for diagnostics.
    pub(super) fn transient_pool(&self) -> &TransientPool {
        &self.transient_pool
    }

    /// Mutable history registry.
    pub(super) fn history_registry_mut(&mut self) -> &mut HistoryRegistry {
        &mut self.history_registry
    }

    /// Mutable transient pool, history registry, and upload arena for graph execution after the
    /// cached graph has been temporarily removed from [`Self::frame_graph_cache`].
    pub(super) fn execution_resources_mut(
        &mut self,
    ) -> (
        &mut TransientPool,
        &mut HistoryRegistry,
        &mut PersistentUploadArena,
        &mut FrameUploadBatchStats,
        &mut Vec<PendingTransientRelease>,
    ) {
        (
            &mut self.transient_pool,
            &mut self.history_registry,
            &mut self.upload_arena,
            &mut self.latest_upload_stats,
            &mut self.pending_transient_releases,
        )
    }

    /// Releases queued transient resources whose submit token has reached `Queue::submit`.
    pub(super) fn release_completed_transient_submits(&mut self, gpu: &crate::gpu::GpuContext) {
        let pending = std::mem::take(&mut self.pending_transient_releases);
        let (completed, waiting) =
            partition_transient_releases(pending, |token| gpu.is_submit_done(token));
        for release in completed {
            for resources in release.resources {
                resources.release_to_pool(&mut self.transient_pool);
            }
        }
        self.pending_transient_releases = waiting;
        self.transient_pool.gc_tick(120);
    }

    /// Latest upload drain stats published by graph execution.
    pub(super) fn latest_upload_stats(&self) -> FrameUploadBatchStats {
        self.latest_upload_stats
    }

    /// Long-lived post-processing resources for main-graph rebuilds.
    pub(super) fn post_processing_resources(&self) -> &MainGraphPostProcessingResources {
        &self.post_processing_resources
    }

    /// Clears persistent upload staging slots when a graph-shape transition invalidates them.
    pub(super) fn reset_upload_arena(&mut self) {
        self.upload_arena.reset();
    }

    /// Returns the upload arena generation for graph-cache reset-policy unit tests.
    #[cfg(test)]
    pub(super) fn upload_arena_generation_for_tests(&self) -> u64 {
        self.upload_arena.next_generation_for_tests()
    }

    /// Synchronizes active view ownership and releases graph-owned view resources immediately.
    pub(super) fn sync_active_views<I>(&mut self, active_views: I) -> Vec<ViewId>
    where
        I: IntoIterator<Item = ViewId>,
    {
        let retired = self.view_resources.sync_active_views(active_views);
        self.release_view_resources(&retired);
        retired
    }

    /// Retires active views matching `predicate` and releases graph-owned resources immediately.
    pub(super) fn retire_views_where(
        &mut self,
        predicate: impl FnMut(ViewId) -> bool,
    ) -> Vec<ViewId> {
        let retired = self.view_resources.retire_where(predicate);
        self.release_view_resources(&retired);
        retired
    }

    /// Releases graph-owned resources for views retired outside the regular active-view registry.
    pub(super) fn release_view_resources(&mut self, retired_views: &[ViewId]) {
        self.frame_graph_cache.release_view_resources(retired_views);
        self.post_processing_resources.retire_views(retired_views);
    }
}

fn partition_transient_releases(
    pending: Vec<PendingTransientRelease>,
    mut is_submit_done: impl FnMut(SubmitToken) -> bool,
) -> (Vec<PendingTransientRelease>, Vec<PendingTransientRelease>) {
    let mut completed = Vec::new();
    let mut waiting = Vec::new();
    for release in pending {
        if is_submit_done(release.token) {
            completed.push(release);
        } else {
            waiting.push(release);
        }
    }
    (completed, waiting)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn auto_exposure_cache_is_stable_across_resource_accesses() {
        let state = RenderGraphState::new();

        let first = state
            .post_processing_resources()
            .auto_exposure_state_cache();
        let second = state
            .post_processing_resources()
            .auto_exposure_state_cache();

        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn transient_release_partition_keeps_incomplete_submits_pending() {
        let pending = vec![
            PendingTransientRelease {
                token: SubmitToken::new(1),
                resources: Vec::new(),
            },
            PendingTransientRelease {
                token: SubmitToken::new(2),
                resources: Vec::new(),
            },
        ];

        let (completed, waiting) = partition_transient_releases(pending, |token| token.raw() <= 1);

        assert_eq!(
            completed
                .iter()
                .map(|release| release.token.raw())
                .collect::<Vec<_>>(),
            vec![1]
        );
        assert_eq!(
            waiting
                .iter()
                .map(|release| release.token.raw())
                .collect::<Vec<_>>(),
            vec![2]
        );
    }
}
