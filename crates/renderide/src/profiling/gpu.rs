/// Requests the GPU features needed for timestamp-query-based profiling.
///
/// Returns the subset of `{TIMESTAMP_QUERY, TIMESTAMP_QUERY_INSIDE_ENCODERS}` that the adapter
/// actually supports. Always queries the adapter regardless of Cargo features so the debug HUD's
/// frame-bracket GPU timing can use real hardware timestamps even in non-Tracy builds; the
/// `tracy`-gated [`GpuProfilerHandle`] consumes the same features for its pass-level path.
///
/// Call this in [`crate::gpu::context`]'s feature-intersection helpers and OR the result into
/// the device's requested features. `TIMESTAMP_QUERY` alone is enough for pass-level profiling;
/// `TIMESTAMP_QUERY_INSIDE_ENCODERS` unlocks encoder-level queries on adapters that offer it,
/// which is what the frame-bracket writes use to surround the entire tick of work.
pub fn timestamp_query_features_if_supported(adapter: &wgpu::Adapter) -> wgpu::Features {
    let needed = wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
    adapter.features() & needed
}

// ---------------------------------------------------------------------------
// PhaseQuery -- GPU timestamp query token, with a no-op stub when `tracy` is off
// ---------------------------------------------------------------------------

/// GPU timestamp query token returned by [`GpuProfilerHandle::begin_query`] /
/// [`GpuProfilerHandle::begin_pass_query`].
///
/// When the `tracy` feature is on this is [`wgpu_profiler::GpuProfilerQuery`]; when it is off
/// this is a zero-sized placeholder so call sites compile identically under both states.
#[cfg(feature = "tracy")]
pub type PhaseQuery = wgpu_profiler::GpuProfilerQuery;

/// Zero-sized placeholder for [`wgpu_profiler::GpuProfilerQuery`] when the `tracy` feature is off.
#[cfg(not(feature = "tracy"))]
pub struct PhaseQuery;

/// One resolved GPU pass timing, flattened from the `wgpu-profiler` result tree.
///
/// Emitted once per frame by [`GpuProfilerHandle::process_finished_frame`] so consumers can
/// display the per-pass breakdown without depending on `wgpu_profiler`'s types or feature gates.
#[derive(Clone, Debug)]
pub struct GpuPassEntry {
    /// Pass label captured at `begin_query` / `begin_pass_query` time.
    pub name: String,
    /// Measured GPU time in milliseconds for this pass.
    pub ms: f32,
    /// Depth in the original query tree (0 for top-level scopes, >0 for nested ones).
    pub depth: u32,
}

/// Per-frame GPU profiler accounting paired with a resolved timestamp result tree.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuProfilerFrameStats {
    /// Monotonic profiler-frame order assigned when the frame was accepted by the profiler.
    pub frame_order: u64,
    /// Timestamp query scopes opened for the frame.
    pub opened_queries: u32,
    /// Lower-priority timestamp scopes skipped for the frame after hitting the query budget.
    pub skipped_queries: u32,
    /// Soft per-frame query budget used to flag unexpectedly dense instrumentation.
    pub soft_query_budget: u32,
}

/// Latest resolved GPU-profiler frame shown by diagnostics surfaces.
#[derive(Clone, Debug, Default)]
pub struct GpuProfilerSnapshot {
    /// Flattened timestamp query tree for the resolved frame.
    pub entries: Vec<GpuPassEntry>,
    /// Query accounting for the resolved frame.
    pub stats: GpuProfilerFrameStats,
}

/// Reads the render-pass timestamp writes reserved for a pass-level query.
///
/// Forwards to [`wgpu_profiler::GpuProfilerQuery::render_pass_timestamp_writes`] when the
/// `tracy` feature is on; returns [`None`] otherwise. Feed the result into
/// [`wgpu::RenderPassDescriptor::timestamp_writes`] when opening the pass, then pair the query
/// with [`GpuProfilerHandle::end_query`] after the pass drops.
#[inline]
pub fn render_pass_timestamp_writes(
    query: Option<&PhaseQuery>,
) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
    #[cfg(feature = "tracy")]
    {
        query.and_then(wgpu_profiler::GpuProfilerQuery::render_pass_timestamp_writes)
    }
    #[cfg(not(feature = "tracy"))]
    {
        let _ = query;
        None
    }
}

/// Reads the compute-pass timestamp writes reserved for a pass-level query.
///
/// Forwards to [`wgpu_profiler::GpuProfilerQuery::compute_pass_timestamp_writes`] when the
/// `tracy` feature is on; returns [`None`] otherwise. Feed the result into
/// [`wgpu::ComputePassDescriptor::timestamp_writes`] when opening the pass, then pair the query
/// with [`GpuProfilerHandle::end_query`] after the pass drops.
#[inline]
pub fn compute_pass_timestamp_writes(
    query: Option<&PhaseQuery>,
) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
    #[cfg(feature = "tracy")]
    {
        query.and_then(wgpu_profiler::GpuProfilerQuery::compute_pass_timestamp_writes)
    }
    #[cfg(not(feature = "tracy"))]
    {
        let _ = query;
        None
    }
}
