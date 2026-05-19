//! Per-frame GPU resource creation counters for Tracy plots.
//!
//! The kind enum and the [`note_resource_churn!`] macro are shared between feature
//! configurations. The real counter state and plot emission live in [`imp`] when the `tracy`
//! feature is on, and in [`stub`] when it is off -- mirroring the
//! [`super::gpu_profiler_impl`] / [`super::gpu_profiler_stub`] split used by the GPU profiler.

#[cfg(feature = "tracy")]
mod imp;
#[cfg(not(feature = "tracy"))]
mod stub;

#[cfg(feature = "tracy")]
pub(crate) use imp::{ResourceChurnSite, flush_resource_churn_plots};
#[cfg(not(feature = "tracy"))]
pub(crate) use stub::{ResourceChurnSite, flush_resource_churn_plots};

/// GPU resource kind counted by the resource-churn profiler.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ResourceChurnKind {
    /// `wgpu::Buffer` creation.
    Buffer,
    /// `wgpu::BindGroup` creation.
    BindGroup,
    /// `wgpu::Texture` creation.
    Texture,
    /// `wgpu::TextureView` creation.
    TextureView,
    /// `wgpu::Sampler` creation.
    Sampler,
    /// `wgpu::RenderPipeline` creation.
    RenderPipeline,
    /// `wgpu::ComputePipeline` creation.
    ComputePipeline,
}

macro_rules! note_resource_churn {
    ($kind:ident, $site:literal) => {{
        static SITE: $crate::profiling::ResourceChurnSite =
            $crate::profiling::ResourceChurnSite::new(
                $crate::profiling::ResourceChurnKind::$kind,
                $site,
            );
        SITE.note();
    }};
}

pub(crate) use note_resource_churn;
