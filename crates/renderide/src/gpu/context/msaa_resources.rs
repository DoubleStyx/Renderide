//! Multisampled color/depth targets and R32Float resolve temps for desktop and stereo paths.

use super::msaa_tiers::clamp_msaa_request_to_supported;
use super::{MsaaStereoDepthResolveR32, MsaaStereoTargets, MsaaTargets};

/// Ensures a single-sample [`wgpu::TextureFormat::R32Float`] texture for MSAA depth resolve + blit.
pub(super) fn ensure_msaa_depth_resolve_r32_view<'a>(
    device: &wgpu::Device,
    extent: (u32, u32),
    cache: &'a mut Option<(wgpu::Texture, wgpu::TextureView)>,
    cache_extent: &mut (u32, u32),
) -> Result<&'a wgpu::TextureView, &'static str> {
    let w = extent.0.max(1);
    let h = extent.1.max(1);
    let needs = *cache_extent != (w, h) || cache.is_none();
    if needs {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("renderide-msaa-depth-resolve-r32"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        *cache_extent = (w, h);
        *cache = Some((tex, view));
    }
    cache
        .as_ref()
        .map(|(_, v)| v)
        .ok_or("msaa depth resolve r32 missing after ensure")
}

/// Ensures multisampled color/depth targets for the main surface; returns [`None`] when `requested_samples` ≤ 1.
pub(super) fn ensure_msaa_targets<'a>(
    device: &wgpu::Device,
    extent: (u32, u32),
    supported: &[u32],
    cache: &'a mut Option<MsaaTargets>,
    requested_samples: u32,
    color_format: wgpu::TextureFormat,
) -> Option<&'a MsaaTargets> {
    let sc = clamp_msaa_request_to_supported(requested_samples, supported);
    if sc <= 1 {
        *cache = None;
        return None;
    }
    let w = extent.0.max(1);
    let h = extent.1.max(1);
    let needs = cache.as_ref().is_none_or(|m| {
        m.extent != (w, h) || m.sample_count != sc || m.color_format != color_format
    });
    if needs {
        let color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("renderide-msaa-color"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: sc,
            dimension: wgpu::TextureDimension::D2,
            format: color_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("renderide-msaa-depth"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: sc,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        *cache = Some(MsaaTargets {
            color_texture,
            color_view,
            depth_texture,
            depth_view,
            sample_count: sc,
            extent: (w, h),
            color_format,
        });
    }
    cache.as_ref()
}

/// Ensures 2-layer (D2Array) multisampled color/depth targets for the OpenXR stereo path.
pub(super) fn ensure_msaa_stereo_targets<'a>(
    device: &wgpu::Device,
    extent: (u32, u32),
    supported_stereo: &[u32],
    cache: &'a mut Option<MsaaStereoTargets>,
    requested_samples: u32,
    color_format: wgpu::TextureFormat,
) -> Option<&'a MsaaStereoTargets> {
    let sc = clamp_msaa_request_to_supported(requested_samples, supported_stereo);
    if sc <= 1 {
        *cache = None;
        return None;
    }
    let w = extent.0.max(1);
    let h = extent.1.max(1);
    let needs = cache.as_ref().is_none_or(|m| {
        m.extent != (w, h) || m.sample_count != sc || m.color_format != color_format
    });
    if needs {
        let size = wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 2,
        };
        let color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("renderide-msaa-color-stereo"),
            size,
            mip_level_count: 1,
            sample_count: sc,
            dimension: wgpu::TextureDimension::D2,
            format: color_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("renderide-msaa-color-stereo-array"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            array_layer_count: Some(2),
            ..Default::default()
        });

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("renderide-msaa-depth-stereo"),
            size,
            mip_level_count: 1,
            sample_count: sc,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("renderide-msaa-depth-stereo-array"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            array_layer_count: Some(2),
            ..Default::default()
        });
        let depth_layer_views = [0u32, 1u32].map(|layer| {
            depth_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("renderide-msaa-depth-stereo-layer"),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: layer,
                array_layer_count: Some(1),
                ..Default::default()
            })
        });

        *cache = Some(MsaaStereoTargets {
            color_texture,
            color_view,
            depth_texture,
            depth_view,
            depth_layer_views,
            sample_count: sc,
            extent: (w, h),
            color_format,
        });
    }
    cache.as_ref()
}

/// Ensures a 2-layer [`wgpu::TextureFormat::R32Float`] temp for stereo MSAA depth resolve.
pub(super) fn ensure_msaa_stereo_depth_resolve<'a>(
    device: &wgpu::Device,
    extent: (u32, u32),
    cache: &'a mut Option<MsaaStereoDepthResolveR32>,
) -> Option<&'a MsaaStereoDepthResolveR32> {
    let w = extent.0.max(1);
    let h = extent.1.max(1);
    let needs = cache.as_ref().is_none_or(|r| r.extent != (w, h));
    if needs {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("renderide-msaa-depth-resolve-r32-stereo"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 2,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let array_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("renderide-msaa-depth-resolve-r32-stereo-array"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            array_layer_count: Some(2),
            ..Default::default()
        });
        let layer_views = [0u32, 1u32].map(|layer| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("renderide-msaa-depth-resolve-r32-stereo-layer"),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: layer,
                array_layer_count: Some(1),
                ..Default::default()
            })
        });
        *cache = Some(MsaaStereoDepthResolveR32 {
            texture,
            array_view,
            layer_views,
            extent: (w, h),
        });
    }
    cache.as_ref()
}
