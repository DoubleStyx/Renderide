//! Lazy OpenXR stereo swapchain and matching depth texture allocation.

use crate::gpu::GpuContext;
use crate::xr::{XR_VIEW_COUNT, XrStereoSwapchain, create_stereo_depth_texture};

use super::types::XrSessionBundle;

/// Creates the lazy stereo swapchain on first successful HMD path.
pub(super) fn ensure_stereo_swapchain(bundle: &mut XrSessionBundle) -> bool {
    profiling::scope!("xr::ensure_stereo_swapchain");
    if bundle.stereo_swapchain.is_some() {
        return true;
    }
    let handles = &bundle.handles;
    let sys_id = handles.xr_system_id;
    let session = handles.xr_session.xr_vulkan_session();
    let inst = handles.xr_session.xr_instance();
    let dev = handles.device.as_ref();
    // SAFETY: `dev` is the wgpu device that was constructed in `xr::bootstrap` from the same
    // `VkDevice` that owns `session`, so the `VkDevice` identity required by `XrStereoSwapchain`
    // holds.
    let res = unsafe { XrStereoSwapchain::new(session, inst, sys_id, dev) };
    match res {
        Ok(sc) => {
            logger::info!(
                "OpenXR swapchain {}x{} (stereo array)",
                sc.resolution.0,
                sc.resolution.1
            );
            bundle.stereo_swapchain = Some(sc);
            true
        }
        Err(e) => {
            logger::debug!("OpenXR swapchain not created: {e}");
            false
        }
    }
}

/// Resizes the wgpu depth texture when the swapchain resolution or layer count changes.
pub(super) fn ensure_stereo_depth_texture(
    gpu: &GpuContext,
    bundle: &mut XrSessionBundle,
    extent: (u32, u32),
) -> bool {
    profiling::scope!("xr::ensure_stereo_depth_texture");
    let need_new_depth = bundle.stereo_depth.as_ref().is_none_or(|(tex, _)| {
        tex.size().width != extent.0
            || tex.size().height != extent.1
            || tex.size().depth_or_array_layers != XR_VIEW_COUNT
    });
    if need_new_depth {
        let limits = gpu.limits().clone();
        bundle.stereo_depth = create_stereo_depth_texture(gpu.device().as_ref(), &limits, extent);
    }
    bundle.stereo_depth.is_some()
}
