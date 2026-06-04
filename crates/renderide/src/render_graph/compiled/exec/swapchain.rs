//! Late swapchain acquisition and prepared-work-item attachment for graph execution.

use crate::gpu::GpuContext;
use crate::render_graph::swapchain_scope::{SwapchainEnterOutcome, SwapchainScope};

use super::super::super::error::GraphExecuteError;
use super::super::{CompiledRenderGraph, FrameView, FrameViewTarget};
use super::types::PerViewWorkItem;

impl CompiledRenderGraph {
    /// Enters [`SwapchainScope`] for `views` if any target the swapchain; `Ok(None)` signals a frame skip.
    ///
    /// The scope holds the [`wgpu::SurfaceTexture`] for the entire frame. After all encoders
    /// are finished, the texture is taken out of the scope via
    /// [`crate::render_graph::swapchain_scope::SwapchainScope::take_surface_texture`] and handed
    /// to the driver thread for `Queue::submit` + `SurfaceTexture::present`. On any early return
    /// before the handoff, the scope still presents on drop so the wgpu Vulkan acquire semaphore
    /// is returned to the pool.
    pub(in crate::render_graph::compiled::exec) fn late_acquire_swapchain_for_prepared_views(
        &self,
        gpu: &mut GpuContext,
        views: &[FrameView<'_>],
        work_items: &mut [PerViewWorkItem],
    ) -> Result<Option<(SwapchainScope, Option<wgpu::TextureView>)>, GraphExecuteError> {
        let acquired = {
            profiling::scope!("graph::late_swapchain_acquire");
            self.enter_swapchain_scope_for_views(gpu, views)?
        };
        let Some((scope, backbuffer_view)) = acquired else {
            return Ok(None);
        };
        Self::attach_swapchain_backbuffer_to_work_items(work_items, backbuffer_view.as_ref())?;
        Ok(Some((scope, backbuffer_view)))
    }

    fn enter_swapchain_scope_for_views(
        &self,
        gpu: &mut GpuContext,
        views: &[FrameView<'_>],
    ) -> Result<Option<(SwapchainScope, Option<wgpu::TextureView>)>, GraphExecuteError> {
        let needs_swapchain = views
            .iter()
            .any(|v| matches!(v.target, FrameViewTarget::Swapchain));
        match SwapchainScope::enter(needs_swapchain, self.needs_surface_acquire, gpu)? {
            SwapchainEnterOutcome::NotNeeded => Ok(Some((SwapchainScope::none(), None))),
            SwapchainEnterOutcome::SkipFrame => Ok(None),
            SwapchainEnterOutcome::Acquired(scope) => {
                let bb = scope.backbuffer_view().cloned();
                Ok(Some((scope, bb)))
            }
        }
    }

    /// Installs the late-acquired swapchain view into prepared per-view work items.
    fn attach_swapchain_backbuffer_to_work_items(
        work_items: &mut [PerViewWorkItem],
        backbuffer_view: Option<&wgpu::TextureView>,
    ) -> Result<(), GraphExecuteError> {
        if !work_items.iter().any(|item| item.target_is_swapchain) {
            return Ok(());
        }
        let Some(backbuffer_view) = backbuffer_view else {
            return Err(GraphExecuteError::MissingSwapchainView);
        };
        for work_item in work_items
            .iter_mut()
            .filter(|item| item.target_is_swapchain)
        {
            work_item.resolved.attach_backbuffer(backbuffer_view);
        }
        Ok(())
    }
}
