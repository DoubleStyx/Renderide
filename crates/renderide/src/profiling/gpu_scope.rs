//! Small GPU-profiler scope helpers for encoder-level command regions.

use super::{GpuProfilerHandle, PhaseQuery};

/// Encoder-level timestamp query scope.
///
/// Use this for copies, clears, resolves, and other command-encoder work that is not naturally
/// covered by a render or compute pass descriptor's timestamp writes. The scope is explicit
/// rather than `Drop`-based because closing a GPU query requires a mutable command encoder.
pub(crate) struct GpuEncoderScope<'a> {
    /// Profiler that owns the query token.
    profiler: Option<&'a GpuProfilerHandle>,
    /// Active query token, if profiling is enabled and the adapter supports encoder timestamps.
    query: Option<PhaseQuery>,
}

impl<'a> GpuEncoderScope<'a> {
    /// Opens an encoder-level timestamp query for `label`.
    pub(crate) fn begin(
        profiler: Option<&'a GpuProfilerHandle>,
        label: impl Into<String>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Self {
        let query = profiler.map(|profiler| profiler.begin_query(label, encoder));
        Self { profiler, query }
    }

    /// Closes the timestamp query, if one was opened.
    pub(crate) fn end(self, encoder: &mut wgpu::CommandEncoder) {
        if let (Some(profiler), Some(query)) = (self.profiler, self.query) {
            profiler.end_query(encoder, query);
        }
    }
}
