//! Reusable lifecycle helpers for backend-owned nonblocking GPU jobs.
//!
//! These helpers are for one-off backend compute or copy work that needs completion
//! notification or a CPU readback outside the render graph. Frame-shape render work stays in
//! the graph so transient resources, barriers, and pass ordering remain explicit there.

mod readback;
mod submit;

pub(crate) use readback::{GpuReadbackJobs, GpuReadbackOutcomes, SubmittedReadbackJob};
pub(crate) use submit::{GpuSubmitJobTracker, SubmittedGpuJob};

/// GPU resources retained until an asynchronous backend job is known to be complete.
///
/// Fields intentionally keep ownership only; many jobs do not need to read them after
/// submission, but the handles must remain alive until the driver has consumed the commands.
#[derive(Default)]
pub(crate) struct GpuJobResources {
    /// Buffers retained until the job completes.
    _buffers: Vec<wgpu::Buffer>,
    /// Textures retained until the job completes.
    _textures: Vec<wgpu::Texture>,
    /// Texture views retained until the job completes.
    _texture_views: Vec<wgpu::TextureView>,
    /// Samplers retained until the job completes.
    _samplers: Vec<wgpu::Sampler>,
    /// Bind groups retained until the job completes.
    _bind_groups: Vec<wgpu::BindGroup>,
}

impl GpuJobResources {
    /// Creates an empty retained-resource set.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Retains one buffer.
    pub(crate) fn with_buffer(mut self, buffer: wgpu::Buffer) -> Self {
        self._buffers.push(buffer);
        self
    }

    /// Retains multiple buffers.
    pub(crate) fn with_buffers(mut self, buffers: Vec<wgpu::Buffer>) -> Self {
        self._buffers.extend(buffers);
        self
    }

    /// Retains one bind group.
    pub(crate) fn with_bind_group(mut self, bind_group: wgpu::BindGroup) -> Self {
        self._bind_groups.push(bind_group);
        self
    }
}
