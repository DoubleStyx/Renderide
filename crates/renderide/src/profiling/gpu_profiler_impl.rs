use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use parking_lot::Mutex;
use wgpu_profiler::{GpuProfiler, GpuProfilerSettings};

use super::PhaseQuery;

/// Number of GPU profiler frames allowed to wait for readback before `wgpu-profiler` starts
/// dropping older timing data.
const GPU_PROFILER_PENDING_FRAMES: usize = 8;
/// Soft per-frame query budget used to warn when instrumentation becomes unusually dense.
const GPU_PROFILER_SOFT_QUERY_BUDGET: u32 = 512;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TracyBridgeMode {
    Unbridged,
    Bridged,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TracyBridgeAction {
    Keep,
    Defer,
    Rebuild(TracyBridgeMode),
}

/// Wraps [`GpuProfiler`] and provides a GPU timestamp query interface for render and
/// compute passes, bridging results to the Tracy GPU timeline.
///
/// Created via [`GpuProfilerHandle::try_new`]; only available when the `tracy` feature is on.
pub struct GpuProfilerHandle {
    /// Underlying query allocator, resolver, readback processor, and Tracy bridge.
    inner: GpuProfiler,
    /// Whether `inner` currently owns a Tracy GPU context and emits Tracy GPU events.
    tracy_bridge_mode: TracyBridgeMode,
    /// Whether any query was opened since the previous successful profiler frame boundary.
    queries_opened_since_frame_end: AtomicBool,
    /// Number of query scopes opened since the previous profiler frame boundary.
    query_count_since_frame_end: AtomicU32,
    /// Per-frame query accounting waiting for the matching resolved timestamp tree.
    pending_frame_stats: Mutex<VecDeque<super::GpuProfilerFrameStats>>,
    /// Whether an over-budget warning has already been logged for the current dense run.
    warned_over_soft_budget: AtomicBool,
}

impl GpuProfilerHandle {
    /// Creates a new handle if the device supports [`wgpu::Features::TIMESTAMP_QUERY`].
    ///
    /// Connects to the Tracy GPU timeline only when a Tracy GUI is already attached. In
    /// `ondemand` mode GPU events emitted while no GUI is attached can cross a later connection
    /// boundary with stale query ids, so late attach is handled by [`Self::refresh_tracy_bridge`]
    /// at clean frame boundaries instead.
    ///
    /// Returns [`None`] when timestamp queries are unavailable; callers fall back to CPU-only
    /// spans without any GPU timeline data.
    pub fn try_new(
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Option<Self> {
        let features = device.features();
        if !features.contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        let backend = adapter.get_info().backend;
        let initial_mode = if tracy_client::Client::is_connected() {
            TracyBridgeMode::Bridged
        } else {
            TracyBridgeMode::Unbridged
        };
        let (inner, tracy_bridge_mode) =
            match create_profiler_for_mode(initial_mode, backend, device, queue) {
                Ok(inner) => (inner, initial_mode),
                Err(e) if initial_mode == TracyBridgeMode::Bridged => {
                    logger::warn!(
                        "GPU profiler Tracy bridge creation failed: {e}; falling back to unbridged"
                    );
                    match create_profiler_for_mode(
                        TracyBridgeMode::Unbridged,
                        backend,
                        device,
                        queue,
                    ) {
                        Ok(inner) => (inner, TracyBridgeMode::Unbridged),
                        Err(e2) => {
                            logger::warn!(
                                "GPU profiler creation failed: {e2}; GPU timeline unavailable"
                            );
                            return None;
                        }
                    }
                }
                Err(e) => {
                    logger::warn!("GPU profiler creation failed: {e}; GPU timeline unavailable");
                    return None;
                }
            };
        Some(Self {
            inner,
            tracy_bridge_mode,
            queries_opened_since_frame_end: AtomicBool::new(false),
            query_count_since_frame_end: AtomicU32::new(0),
            pending_frame_stats: Mutex::new(VecDeque::new()),
            warned_over_soft_budget: AtomicBool::new(false),
        })
    }

    /// Rebuilds the Tracy bridge when the GUI connection state changes at a clean frame boundary.
    ///
    /// The underlying `tracy-client` GPU API does not gate serial GPU events on
    /// `TRACY_ON_DEMAND`, so Renderide keeps `wgpu-profiler` unbridged while the GUI is
    /// disconnected. When a GUI connects, this method swaps in a fresh Tracy-bridged profiler
    /// before the next frame opens any queries. When it disconnects, the bridge is replaced by an
    /// unbridged profiler for the same reason.
    pub fn refresh_tracy_bridge(
        &mut self,
        backend: wgpu::Backend,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pending_submit_end: bool,
    ) {
        let target_mode = if tracy_client::Client::is_connected() {
            TracyBridgeMode::Bridged
        } else {
            TracyBridgeMode::Unbridged
        };
        let action = tracy_bridge_action(
            self.tracy_bridge_mode,
            target_mode,
            self.has_queries_opened_since_frame_end(),
            pending_submit_end,
        );
        let TracyBridgeAction::Rebuild(mode) = action else {
            return;
        };
        match create_profiler_for_mode(mode, backend, device, queue) {
            Ok(inner) => {
                self.replace_inner(inner, mode);
                match mode {
                    TracyBridgeMode::Bridged => {
                        logger::info!("GPU profiler Tracy bridge enabled");
                    }
                    TracyBridgeMode::Unbridged => {
                        logger::info!("GPU profiler Tracy bridge disabled");
                    }
                }
            }
            Err(e) if mode == TracyBridgeMode::Bridged => {
                logger::warn!(
                    "GPU profiler Tracy bridge creation failed: {e}; keeping unbridged profiler"
                );
            }
            Err(e) => {
                logger::warn!(
                    "GPU profiler unbridged rebuild failed after Tracy disconnect: {e}; preserving current profiler"
                );
            }
        }
    }

    /// Replaces the underlying profiler and clears accounting that belongs to the old query ids.
    fn replace_inner(&mut self, inner: GpuProfiler, tracy_bridge_mode: TracyBridgeMode) {
        self.inner = inner;
        self.tracy_bridge_mode = tracy_bridge_mode;
        self.queries_opened_since_frame_end
            .store(false, Ordering::Release);
        self.query_count_since_frame_end.store(0, Ordering::Release);
        self.pending_frame_stats.lock().clear();
        self.warned_over_soft_budget.store(false, Ordering::Release);
    }

    /// Marks the active profiler frame as non-empty.
    #[inline]
    fn note_query_opened(&self) {
        self.queries_opened_since_frame_end
            .store(true, Ordering::Release);
        self.query_count_since_frame_end
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Returns whether the current profiler frame has opened any GPU queries.
    #[inline]
    pub fn has_queries_opened_since_frame_end(&self) -> bool {
        self.queries_opened_since_frame_end.load(Ordering::Acquire)
    }

    /// Opens an encoder-level GPU timestamp query.
    ///
    /// Writes `WriteTimestamp` commands into `encoder` -- requires
    /// [`wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS`]. If the adapter lacks that
    /// feature the query is silently a no-op. Prefer [`Self::begin_pass_query`] for
    /// individual passes. The returned [`PhaseQuery`] must be closed via [`Self::end_query`]
    /// before [`Self::resolve_queries`] is called.
    #[inline]
    pub fn begin_query(
        &self,
        label: impl Into<String>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> PhaseQuery {
        self.note_query_opened();
        self.inner.begin_query(label, encoder)
    }

    /// Reserves a pass-level timestamp query for a single render or compute pass.
    ///
    /// The returned [`PhaseQuery`] carries `timestamp_writes` the caller must inject into the
    /// [`wgpu::RenderPassDescriptor`] / [`wgpu::ComputePassDescriptor`] via
    /// [`super::render_pass_timestamp_writes`] or [`super::compute_pass_timestamp_writes`].
    /// After the pass drops, close the query with [`Self::end_query`]. Requires only
    /// [`wgpu::Features::TIMESTAMP_QUERY`].
    #[inline]
    pub fn begin_pass_query(
        &self,
        label: impl Into<String>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> PhaseQuery {
        self.note_query_opened();
        self.inner.begin_pass_query(label, encoder)
    }

    /// Closes a query previously opened with [`Self::begin_query`] or
    /// [`Self::begin_pass_query`].
    #[inline]
    pub fn end_query(&self, encoder: &mut wgpu::CommandEncoder, query: PhaseQuery) {
        self.inner.end_query(encoder, query);
    }

    /// Inserts query-resolve commands into `encoder` for all unresolved queries this frame.
    ///
    /// Call once per encoder just before [`wgpu::CommandEncoder::finish`]. The encoder used
    /// for resolution must be submitted **after** all encoders that opened queries in this
    /// profiling frame.
    #[inline]
    pub fn resolve_queries(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.inner.resolve_queries(encoder);
    }

    /// Marks the end of the current profiling frame only if at least one query was opened.
    ///
    /// Call once per render tick after all command encoders for this frame have been submitted.
    /// Empty CPU ticks are intentionally ignored so `wgpu-profiler` does not enqueue empty GPU
    /// frames that later appear as missing markers in Tracy.
    #[inline]
    pub fn end_frame_if_queries_opened(&mut self) -> bool {
        let had_queries = self
            .queries_opened_since_frame_end
            .swap(false, Ordering::AcqRel);
        let opened_queries = self.query_count_since_frame_end.swap(0, Ordering::AcqRel);
        if had_queries {
            match self.inner.end_frame() {
                Ok(()) => {
                    self.record_frame_stats(opened_queries);
                    self.warn_if_soft_budget_exceeded(opened_queries);
                }
                Err(e) => {
                    logger::warn!("GPU profiler end_frame failed: {e}");
                }
            }
        }
        had_queries
    }

    /// Stores query accounting for a frame accepted by `wgpu-profiler`.
    fn record_frame_stats(&self, opened_queries: u32) {
        let mut pending = self.pending_frame_stats.lock();
        if pending.len() >= GPU_PROFILER_PENDING_FRAMES {
            pending.pop_front();
        }
        pending.push_back(super::GpuProfilerFrameStats {
            opened_queries,
            skipped_queries: 0,
            soft_query_budget: GPU_PROFILER_SOFT_QUERY_BUDGET,
        });
    }

    /// Logs the first over-budget query frame until query density drops below the soft budget.
    fn warn_if_soft_budget_exceeded(&self, opened_queries: u32) {
        if opened_queries <= GPU_PROFILER_SOFT_QUERY_BUDGET {
            self.warned_over_soft_budget.store(false, Ordering::Release);
            return;
        }
        if !self.warned_over_soft_budget.swap(true, Ordering::AcqRel) {
            logger::warn!(
                "GPU profiler opened {opened_queries} timestamp queries in one frame; soft budget is {GPU_PROFILER_SOFT_QUERY_BUDGET}"
            );
        }
    }

    /// Drains results from the oldest completed profiling frame into Tracy and returns a
    /// flattened list of per-pass timings.
    ///
    /// Call once per render tick after [`Self::end_frame_if_queries_opened`]. Results are
    /// available 1-2 frames after recording because the GPU needs to finish executing before
    /// the timestamps are readable. `timestamp_period` is from
    /// [`wgpu::Queue::get_timestamp_period`].
    ///
    /// Returns [`None`] when no frame has completed yet or when `wgpu_profiler` could not
    /// resolve the frame's timestamps. Otherwise returns a depth-annotated preorder traversal
    /// of the query tree so callers can render it as a flat table.
    #[inline]
    pub fn process_finished_frame(
        &mut self,
        timestamp_period: f32,
    ) -> Option<super::GpuProfilerSnapshot> {
        let tree = self.inner.process_finished_frame(timestamp_period)?;
        let mut out = Vec::new();
        flatten_results(&tree, 0, &mut out);
        let stats =
            self.pending_frame_stats
                .lock()
                .pop_front()
                .unwrap_or(super::GpuProfilerFrameStats {
                    opened_queries: out.len() as u32,
                    skipped_queries: 0,
                    soft_query_budget: GPU_PROFILER_SOFT_QUERY_BUDGET,
                });
        Some(super::GpuProfilerSnapshot {
            entries: out,
            stats,
        })
    }
}

/// Preorder-flattens a [`wgpu_profiler::GpuTimerQueryResult`] tree into
/// [`super::GpuPassEntry`] rows. Skips entries with no timing data (queries that were never
/// written, e.g. when timestamp writes were not consumed by a pass).
fn flatten_results(
    nodes: &[wgpu_profiler::GpuTimerQueryResult],
    depth: u32,
    out: &mut Vec<super::GpuPassEntry>,
) {
    for node in nodes {
        if let Some(range) = node.time.as_ref() {
            let ms = ((range.end - range.start) * 1000.0) as f32;
            out.push(super::GpuPassEntry {
                name: node.label.clone(),
                ms,
                depth,
            });
        }
        flatten_results(&node.nested_queries, depth + 1, out);
    }
}

fn profiler_settings() -> GpuProfilerSettings {
    GpuProfilerSettings {
        enable_timer_queries: true,
        enable_debug_groups: true,
        max_num_pending_frames: GPU_PROFILER_PENDING_FRAMES,
    }
}

fn create_profiler_for_mode(
    mode: TracyBridgeMode,
    backend: wgpu::Backend,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<GpuProfiler, wgpu_profiler::CreationError> {
    let settings = profiler_settings();
    match mode {
        TracyBridgeMode::Unbridged => GpuProfiler::new(device, settings),
        TracyBridgeMode::Bridged => {
            GpuProfiler::new_with_tracy_client(settings, backend, device, queue)
        }
    }
}

fn tracy_bridge_action(
    current_mode: TracyBridgeMode,
    target_mode: TracyBridgeMode,
    frame_has_queries: bool,
    pending_submit_end: bool,
) -> TracyBridgeAction {
    if current_mode == target_mode {
        return TracyBridgeAction::Keep;
    }
    if frame_has_queries || pending_submit_end {
        return TracyBridgeAction::Defer;
    }
    TracyBridgeAction::Rebuild(target_mode)
}

#[cfg(test)]
mod tests {
    use super::{TracyBridgeAction, TracyBridgeMode, tracy_bridge_action};

    #[test]
    fn bridge_action_keeps_current_mode_when_connection_matches() {
        assert_eq!(
            tracy_bridge_action(
                TracyBridgeMode::Unbridged,
                TracyBridgeMode::Unbridged,
                false,
                false,
            ),
            TracyBridgeAction::Keep
        );
        assert_eq!(
            tracy_bridge_action(
                TracyBridgeMode::Bridged,
                TracyBridgeMode::Bridged,
                false,
                false,
            ),
            TracyBridgeAction::Keep
        );
    }

    #[test]
    fn bridge_action_rebuilds_for_late_connect_at_clean_boundary() {
        assert_eq!(
            tracy_bridge_action(
                TracyBridgeMode::Unbridged,
                TracyBridgeMode::Bridged,
                false,
                false,
            ),
            TracyBridgeAction::Rebuild(TracyBridgeMode::Bridged)
        );
    }

    #[test]
    fn bridge_action_rebuilds_for_disconnect_at_clean_boundary() {
        assert_eq!(
            tracy_bridge_action(
                TracyBridgeMode::Bridged,
                TracyBridgeMode::Unbridged,
                false,
                false,
            ),
            TracyBridgeAction::Rebuild(TracyBridgeMode::Unbridged)
        );
    }

    #[test]
    fn bridge_action_defers_when_frame_has_queries() {
        assert_eq!(
            tracy_bridge_action(
                TracyBridgeMode::Unbridged,
                TracyBridgeMode::Bridged,
                true,
                false,
            ),
            TracyBridgeAction::Defer
        );
    }

    #[test]
    fn bridge_action_defers_when_submit_end_is_pending() {
        assert_eq!(
            tracy_bridge_action(
                TracyBridgeMode::Bridged,
                TracyBridgeMode::Unbridged,
                false,
                true,
            ),
            TracyBridgeAction::Defer
        );
    }
}
