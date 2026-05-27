use super::PhaseQuery;

/// Zero-sized stub that stands in for the real GPU profiler handle when the `tracy` feature
/// is not enabled. All methods are no-ops inlined to nothing; the stub is never instantiated
/// because [`GpuProfilerHandle::try_new`] always returns [`None`].
pub struct GpuProfilerHandle;

impl GpuProfilerHandle {
    /// Always returns [`None`]; GPU profiling is unavailable without the `tracy` feature.
    #[inline]
    pub fn try_new(
        _adapter: &wgpu::Adapter,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) -> Option<Self> {
        None
    }

    /// No-op stub; Tracy bridge state only exists when the `tracy` feature is enabled.
    #[inline]
    pub fn refresh_tracy_bridge(
        &self,
        _backend: wgpu::Backend,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _pending_submit_end: bool,
    ) {
    }

    /// No-op stub; see the `tracy` feature variant for the real implementation.
    #[inline]
    pub fn begin_query(
        &self,
        _label: impl Into<String>,
        _encoder: &mut wgpu::CommandEncoder,
    ) -> PhaseQuery {
        PhaseQuery
    }

    /// No-op stub; see the `tracy` feature variant for the real implementation.
    #[inline]
    pub fn begin_pass_query(
        &self,
        _label: impl Into<String>,
        _encoder: &mut wgpu::CommandEncoder,
    ) -> PhaseQuery {
        PhaseQuery
    }

    /// No-op stub; see the `tracy` feature variant for the real implementation.
    #[inline]
    pub fn end_query(&self, _encoder: &mut wgpu::CommandEncoder, _query: PhaseQuery) {}

    /// No-op stub; see the `tracy` feature variant for the real implementation.
    #[inline]
    pub fn resolve_queries(&self, _encoder: &mut wgpu::CommandEncoder) {}

    /// No-op stub; see the `tracy` feature variant for the real implementation.
    #[inline]
    pub fn has_queries_opened_since_frame_end(&self) -> bool {
        false
    }

    /// No-op stub; see the `tracy` feature variant for the real implementation.
    #[inline]
    pub fn end_frame_if_queries_opened(&self) -> bool {
        false
    }

    /// No-op stub; see the `tracy` feature variant for the real implementation.
    ///
    /// Always returns [`None`] because the stub never opens queries.
    #[inline]
    pub fn process_finished_frame(
        &self,
        _timestamp_period: f32,
    ) -> Option<super::GpuProfilerSnapshot> {
        None
    }
}
