//! RAII guard for the swapchain surface texture acquired at the start of a multi-view frame.
//!
//! ## Safety contract
//!
//! wgpu's Vulkan backend ties a per-frame acquire semaphore to the [`wgpu::SurfaceTexture`].
//! The texture **must** be presented -- via [`wgpu::SurfaceTexture::present`] -- regardless of
//! whether the frame completes successfully, returns an error, or panics, so the semaphore and
//! image return to the swapchain pool. Failing to present causes a secondary panic
//! (`SwapchainAcquireSemaphore` still in use) that masks the original failure.
//!
//! [`SwapchainScope`] owns the surface texture and calls `present()` in its [`Drop`]
//! implementation, providing the guarantee unconditionally. This replaces the tuple-field
//! drop-order trick that was previously inlined in
//! `crates/renderide/src/render_graph/compiled/exec.rs`.

use crate::gpu::GpuContext;
use crate::present::{SurfaceAcquireTrace, SurfaceFrameOutcome, acquire_surface_outcome_traced};
use crate::render_graph::error::GraphExecuteError;

/// Outcome of [`SwapchainScope::enter`].
pub enum SwapchainEnterOutcome {
    /// Swapchain surface was not needed (graph does not target the backbuffer).
    NotNeeded,
    /// Frame acquisition was skipped (timeout, occluded, or swapchain reconfigured).
    SkipFrame,
    /// Surface texture acquired; backbuffer view is available in the scope.
    Acquired(SwapchainScope),
}

/// RAII guard that presents the swapchain surface texture when dropped.
///
/// Callers must keep this value alive for the entire duration of the frame.
/// The swapchain image is presented in [`Drop`]; errors during drop are silently swallowed
/// since `present()` has no return value.
pub struct SwapchainScope {
    /// Held for the frame; presented and taken on drop.
    inner: Option<wgpu::SurfaceTexture>,
    /// Pre-created default view for render-pass attachment.
    backbuffer_view: Option<wgpu::TextureView>,
}

impl SwapchainScope {
    /// Creates a no-op scope that performs no present on drop (offscreen-only path).
    pub fn none() -> Self {
        Self {
            inner: None,
            backbuffer_view: None,
        }
    }

    /// Wraps an acquired surface texture and creates its default color view.
    pub fn new(tex: wgpu::SurfaceTexture) -> Self {
        let view = tex
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            inner: Some(tex),
            backbuffer_view: Some(view),
        }
    }

    /// Acquires the swapchain surface when the graph needs it, returning the entry outcome.
    ///
    /// Returns [`SwapchainEnterOutcome::NotNeeded`] when neither `needs_swapchain` nor
    /// `graph_needs_surface_acquire` is true. Headless contexts never acquire a surface -- callers
    /// are responsible for routing swapchain views to offscreen targets before reaching this
    /// point.
    pub fn enter(
        needs_swapchain: bool,
        graph_needs_surface_acquire: bool,
        gpu: &mut GpuContext,
    ) -> Result<SwapchainEnterOutcome, GraphExecuteError> {
        if !needs_swapchain || !graph_needs_surface_acquire {
            return Ok(SwapchainEnterOutcome::NotNeeded);
        }
        if gpu.is_headless() {
            return Err(GraphExecuteError::SwapchainRequiresWindow);
        }
        profiling::scope!("gpu::swapchain_acquire");
        // wgpu holds the invariant that a `Surface` has at most one outstanding
        // `SurfaceTexture` -- the previous frame's `present()` must complete before the next
        // `get_current_texture()` call. Wait specifically on the prior present instead of
        // doing a full `flush_driver` so non-surface batches (e.g. Hi-Z readback submits,
        // `on_submitted_work_done` callbacks) stay pipelined with frame N+1's recording.
        // With `desired_maximum_frame_latency >= 2` (the default) the driver thread can
        // present frame N's image while frame N+1 is still recording, so by the time the
        // main thread reaches this barrier the previous present has typically already
        // completed and the wait is near-instant. With `max_frame_latency = 1` this becomes
        // the dominant frame stall; that tradeoff is documented on
        // `RenderingSettings::max_frame_latency`.
        {
            profiling::scope!("gpu::wait_previous_present.desktop_graph");
            gpu.wait_for_previous_present();
        };
        let outcome = acquire_surface_outcome_traced(gpu, SurfaceAcquireTrace::DesktopGraph)?;
        match outcome {
            SurfaceFrameOutcome::Skip | SurfaceFrameOutcome::Reconfigured => {
                Ok(SwapchainEnterOutcome::SkipFrame)
            }
            SurfaceFrameOutcome::Acquired(tex) => {
                Ok(SwapchainEnterOutcome::Acquired(Self::new(tex)))
            }
        }
    }

    /// Returns a reference to the backbuffer color view, or [`None`] for offscreen-only frames.
    pub fn backbuffer_view(&self) -> Option<&wgpu::TextureView> {
        self.backbuffer_view.as_ref()
    }

    /// Takes the underlying [`wgpu::SurfaceTexture`] out of the scope without presenting it,
    /// handing ownership to the caller.
    ///
    /// Intended for the driver-thread submit path: once the caller hands the texture off to
    /// [`crate::gpu::GpuContext::submit_frame_batch`], the driver thread performs `present()`
    /// after the submit. The scope's [`Drop`] tolerates the texture being gone -- it becomes a
    /// no-op for this frame. The backbuffer view is dropped alongside (it is tied to the
    /// texture's lifetime).
    ///
    /// Returns [`None`] for scopes created via [`Self::none`] (offscreen-only path).
    pub fn take_surface_texture(&mut self) -> Option<wgpu::SurfaceTexture> {
        self.backbuffer_view = None;
        self.inner.take()
    }
}

impl Drop for SwapchainScope {
    fn drop(&mut self) {
        if let Some(tex) = self.inner.take() {
            tex.present();
        }
    }
}
