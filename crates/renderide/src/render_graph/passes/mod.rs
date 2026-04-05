//! Concrete render passes registered on a [`super::CompiledRenderGraph`].
//!
//! Phase 2 can add G-buffer, lighting, post, and UI passes here.

mod swapchain_clear;

pub use swapchain_clear::SwapchainClearPass;
