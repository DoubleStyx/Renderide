//! Swapchain image acquire/release under the shared GPU queue access gate.

use openxr as xr;
use parking_lot::Mutex;

use crate::gpu::GpuContext;

/// Acquires one OpenXR swapchain image while holding the shared Vulkan queue access gate.
///
/// Briefly locks the swapchain mutex around `xrAcquireSwapchainImage`. The lock is dropped
/// before `xrWaitSwapchainImage` so the long compositor wait does not block the driver
/// thread's `xrReleaseSwapchainImage` for an unrelated frame.
pub(super) fn acquire_swapchain_image(
    gpu: &GpuContext,
    swapchain: &Mutex<xr::Swapchain<xr::Vulkan>>,
) -> Result<usize, xr::sys::Result> {
    let _gate = gpu.gpu_queue_access_gate().lock();
    swapchain.lock().acquire_image().map(|i| i as usize)
}

/// Releases one OpenXR swapchain image while holding the shared Vulkan queue access gate.
///
/// Used by the failure recovery paths in `submit.rs` where no finalize work was queued for
/// the driver thread. The success path releases on the driver thread instead, see
/// [`crate::gpu::driver_thread::XrFinalizeKind::Projection`].
pub(super) fn release_swapchain_image(
    gpu: &GpuContext,
    swapchain: &Mutex<xr::Swapchain<xr::Vulkan>>,
) -> Result<(), xr::sys::Result> {
    let _gate = gpu.gpu_queue_access_gate().lock();
    swapchain.lock().release_image()
}
