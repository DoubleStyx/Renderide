//! Pending GPU texture allocation for format-table entries (attach and Texture2D data path).

use std::sync::Arc;

use crate::gpu_pools::{GpuCubemap, GpuRenderTexture, GpuTexture2d, GpuTexture3d};

use super::super::AssetTransferQueue;

/// Ensures [`GpuTexture2d`](crate::gpu_pools::GpuTexture2d) instances exist for every format table entry (called on attach and before data upload).
pub fn flush_pending_texture_allocations(
    queue: &mut AssetTransferQueue,
    device: &Arc<wgpu::Device>,
) {
    let ids: Vec<i32> = queue.texture_formats.keys().copied().collect();
    for id in ids {
        if queue.texture_pool.get_texture(id).is_some() {
            continue;
        }
        let Some(fmt) = queue.texture_formats.get(&id).cloned() else {
            continue;
        };
        let props = queue.texture_properties.get(&id);
        let Some(limits) = queue.gpu_limits.as_ref() else {
            logger::warn!("texture {id}: gpu_limits missing; cannot allocate on attach");
            continue;
        };
        let Some(tex) =
            GpuTexture2d::new_from_format(device.as_ref(), limits.as_ref(), &fmt, props)
        else {
            logger::warn!("texture {id}: failed to allocate GPU texture on attach");
            continue;
        };
        let _ = queue.texture_pool.insert_texture(tex);
    }
}

/// Allocates [`GpuRenderTexture`](crate::gpu_pools::GpuRenderTexture) targets for pending render-texture format entries.
pub fn flush_pending_render_texture_allocations(
    queue: &mut AssetTransferQueue,
    device: &Arc<wgpu::Device>,
) {
    let ids: Vec<i32> = queue.render_texture_formats.keys().copied().collect();
    for id in ids {
        if queue.render_texture_pool.get(id).is_some() {
            continue;
        }
        let Some(fmt) = queue.render_texture_formats.get(&id).cloned() else {
            continue;
        };
        let Some(limits) = queue.gpu_limits.as_ref() else {
            logger::warn!("render texture {id}: gpu_limits missing; cannot allocate on attach");
            continue;
        };
        let Some(tex) = GpuRenderTexture::new_from_format(
            device.as_ref(),
            limits.as_ref(),
            &fmt,
            queue.render_texture_hdr_color,
        ) else {
            logger::warn!("render texture {id}: failed to allocate GPU targets on attach");
            continue;
        };
        let _ = queue.render_texture_pool.insert_texture(tex);
    }
}

/// Ensures [`GpuTexture3d`](crate::gpu_pools::GpuTexture3d) instances exist for pending format table entries.
pub fn flush_pending_texture3d_allocations(
    queue: &mut AssetTransferQueue,
    device: &Arc<wgpu::Device>,
) {
    let ids: Vec<i32> = queue.texture3d_formats.keys().copied().collect();
    for id in ids {
        if queue.texture3d_pool.get_texture(id).is_some() {
            continue;
        }
        let Some(fmt) = queue.texture3d_formats.get(&id).cloned() else {
            continue;
        };
        let props = queue.texture3d_properties.get(&id);
        let Some(limits) = queue.gpu_limits.as_ref() else {
            logger::warn!("texture3d {id}: gpu_limits missing; cannot allocate on attach");
            continue;
        };
        let Some(tex) =
            GpuTexture3d::new_from_format(device.as_ref(), limits.as_ref(), &fmt, props)
        else {
            logger::warn!("texture3d {id}: failed to allocate GPU texture on attach");
            continue;
        };
        let _ = queue.texture3d_pool.insert_texture(tex);
    }
}

/// Ensures [`GpuCubemap`](crate::gpu_pools::GpuCubemap) instances exist for pending format table entries.
pub fn flush_pending_cubemap_allocations(
    queue: &mut AssetTransferQueue,
    device: &Arc<wgpu::Device>,
) {
    let ids: Vec<i32> = queue.cubemap_formats.keys().copied().collect();
    for id in ids {
        if queue.cubemap_pool.get_texture(id).is_some() {
            continue;
        }
        let Some(fmt) = queue.cubemap_formats.get(&id).cloned() else {
            continue;
        };
        let props = queue.cubemap_properties.get(&id);
        let Some(limits) = queue.gpu_limits.as_ref() else {
            logger::warn!("cubemap {id}: gpu_limits missing; cannot allocate on attach");
            continue;
        };
        let Some(tex) = GpuCubemap::new_from_format(device.as_ref(), limits.as_ref(), &fmt, props)
        else {
            logger::warn!("cubemap {id}: failed to allocate GPU texture on attach");
            continue;
        };
        let _ = queue.cubemap_pool.insert_texture(tex);
    }
}
