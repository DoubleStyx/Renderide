//! Lazy OpenXR stereo swapchain and renderer-owned HMD target allocation.

use crate::gpu::{GpuContext, VR_MIRROR_EYE_LAYER};
use crate::xr::{XR_COLOR_FORMAT, XR_VIEW_COUNT, XrStereoSwapchain, create_stereo_depth_texture};

use super::types::{XrOwnedHmdTargets, XrSessionBundle};

/// Creates the lazy stereo swapchain on first successful HMD path.
pub(super) fn ensure_stereo_swapchain(bundle: &mut XrSessionBundle) -> bool {
    profiling::scope!("xr::ensure_stereo_swapchain");
    if bundle.stereo_swapchain.is_some() {
        return true;
    }
    let handles = &bundle.handles;
    let res = XrStereoSwapchain::new(handles);
    match res {
        Ok(sc) => {
            logger::info!(
                "OpenXR swapchain {}x{} (stereo array) runtime_images={}",
                sc.resolution.0,
                sc.resolution.1,
                sc.image_count(),
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

/// Resizes renderer-owned HMD color and depth targets when the swapchain resolution changes.
pub(super) fn ensure_owned_hmd_targets(
    gpu: &GpuContext,
    bundle: &mut XrSessionBundle,
    extent: (u32, u32),
) -> bool {
    profiling::scope!("xr::ensure_owned_hmd_targets");
    let need_new_targets = bundle
        .hmd_targets
        .as_ref()
        .is_none_or(|targets| !targets.matches_extent(extent));
    if need_new_targets {
        let limits = gpu.limits().clone();
        bundle.hmd_targets = create_owned_hmd_targets(gpu.device().as_ref(), &limits, extent);
    }
    bundle.hmd_targets.is_some()
}

/// Creates renderer-owned stereo color/depth targets for one HMD resolution.
fn create_owned_hmd_targets(
    device: &wgpu::Device,
    limits: &crate::gpu::GpuLimits,
    extent: (u32, u32),
) -> Option<XrOwnedHmdTargets> {
    let (w, h) = (extent.0.max(1), extent.1.max(1));
    if !limits.texture_2d_fits(w, h) {
        logger::warn!(
            "xr owned HMD targets: extent {w}x{h} exceeds max_texture_dimension_2d={}; skipping",
            limits.max_texture_dimension_2d()
        );
        return None;
    }
    if !limits.array_layers_fit(XR_VIEW_COUNT) {
        logger::warn!(
            "xr owned HMD targets: requires {XR_VIEW_COUNT} array layers but max_texture_array_layers={}; skipping",
            limits.max_texture_array_layers()
        );
        return None;
    }

    let (depth_texture, depth_view) = create_stereo_depth_texture(device, limits, (w, h))?;
    let color_texture = device.create_texture(&owned_hmd_color_texture_descriptor((w, h)));
    let color_array_view = color_texture.create_view(&owned_hmd_color_array_view_descriptor());
    crate::profiling::note_resource_churn!(TextureView, "xr::owned_hmd_color_array_view");
    let mirror_eye_view = color_texture.create_view(&owned_hmd_mirror_eye_view_descriptor());
    crate::profiling::note_resource_churn!(TextureView, "xr::owned_hmd_mirror_eye_view");
    Some(XrOwnedHmdTargets::new(
        color_texture,
        color_array_view,
        mirror_eye_view,
        depth_texture,
        depth_view,
        (w, h),
    ))
}

/// Texture usages for the renderer-owned HMD color target.
fn owned_hmd_color_texture_usage() -> wgpu::TextureUsages {
    wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING
}

/// Descriptor for the renderer-owned stereo HMD color texture.
fn owned_hmd_color_texture_descriptor(extent: (u32, u32)) -> wgpu::TextureDescriptor<'static> {
    wgpu::TextureDescriptor {
        label: Some("xr_owned_hmd_color"),
        size: wgpu::Extent3d {
            width: extent.0.max(1),
            height: extent.1.max(1),
            depth_or_array_layers: XR_VIEW_COUNT,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: XR_COLOR_FORMAT,
        usage: owned_hmd_color_texture_usage(),
        view_formats: &[],
    }
}

/// Descriptor for the two-layer color view used by the HMD graph.
fn owned_hmd_color_array_view_descriptor() -> wgpu::TextureViewDescriptor<'static> {
    wgpu::TextureViewDescriptor {
        label: Some("xr_owned_hmd_color_array"),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        array_layer_count: Some(XR_VIEW_COUNT),
        ..Default::default()
    }
}

/// Descriptor for the left-eye color view used by the desktop mirror staging pass.
fn owned_hmd_mirror_eye_view_descriptor() -> wgpu::TextureViewDescriptor<'static> {
    wgpu::TextureViewDescriptor {
        label: Some("xr_owned_hmd_mirror_eye"),
        dimension: Some(wgpu::TextureViewDimension::D2),
        base_array_layer: VR_MIRROR_EYE_LAYER,
        array_layer_count: Some(1),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owned_hmd_color_texture_has_stereo_render_and_sample_usage() {
        let desc = owned_hmd_color_texture_descriptor((2496, 2688));

        assert_eq!(desc.size.width, 2496);
        assert_eq!(desc.size.height, 2688);
        assert_eq!(desc.size.depth_or_array_layers, XR_VIEW_COUNT);
        assert_eq!(desc.format, XR_COLOR_FORMAT);
        assert!(desc.usage.contains(wgpu::TextureUsages::RENDER_ATTACHMENT));
        assert!(desc.usage.contains(wgpu::TextureUsages::TEXTURE_BINDING));
        assert!(!desc.usage.contains(wgpu::TextureUsages::COPY_SRC));
        assert!(!desc.usage.contains(wgpu::TextureUsages::COPY_DST));
    }

    #[test]
    fn owned_hmd_color_views_select_array_and_left_eye() {
        let array = owned_hmd_color_array_view_descriptor();
        assert_eq!(array.dimension, Some(wgpu::TextureViewDimension::D2Array));
        assert_eq!(array.array_layer_count, Some(XR_VIEW_COUNT));

        let mirror = owned_hmd_mirror_eye_view_descriptor();
        assert_eq!(mirror.dimension, Some(wgpu::TextureViewDimension::D2));
        assert_eq!(mirror.base_array_layer, VR_MIRROR_EYE_LAYER);
        assert_eq!(mirror.array_layer_count, Some(1));
    }
}
