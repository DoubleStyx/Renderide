//! Texture3D full-volume mip upload wrappers.

use super::TextureUploadError;
use super::region::{TextureRegionWrite, write_texture_region, write_texture_region_with_gate};
use crate::gpu::GpuQueueAccessMode;

/// Descriptor for [`write_texture3d_volume_mip`]: one full 3D subresource write via [`wgpu::Queue::write_texture`].
pub(in crate::assets::texture::upload) struct Texture3dVolumeMipWrite<'a> {
    /// Queue used for the texel copy.
    pub queue: &'a wgpu::Queue,
    /// Shared GPU queue access gate for [`wgpu::Queue::write_texture`]; see
    /// [`crate::gpu::GpuQueueAccessGate`].
    pub gpu_queue_access_gate: &'a crate::gpu::GpuQueueAccessGate,
    /// Queue-gate acquisition policy for this write.
    pub queue_access_mode: GpuQueueAccessMode,
    /// Destination texture.
    pub texture: &'a wgpu::Texture,
    /// Mip level index.
    pub mip_level: u32,
    /// Logical width in texels.
    pub width: u32,
    /// Logical height in texels.
    pub height: u32,
    /// Depth in texels (array layers for 3D).
    pub depth: u32,
    /// Texel format (must match texture creation).
    pub format: wgpu::TextureFormat,
    /// Tightly packed mip bytes for the full volume at `mip_level`.
    pub bytes: &'a [u8],
}

/// Writes one mip level of a 3D texture (full `width` x `height` x `depth` volume).
pub(in crate::assets::texture::upload) fn write_texture3d_volume_mip(
    write: &Texture3dVolumeMipWrite<'_>,
) -> Result<(), TextureUploadError> {
    profiling::scope!("asset::texture3d_write_volume_mip");
    let Texture3dVolumeMipWrite {
        queue,
        gpu_queue_access_gate,
        queue_access_mode,
        texture,
        mip_level,
        width,
        height,
        depth,
        format,
        bytes,
    } = *write;
    write_texture_region(TextureRegionWrite {
        queue,
        gpu_queue_access_gate,
        queue_access_mode,
        destination: wgpu::TexelCopyTextureInfo {
            texture,
            mip_level,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        width,
        height,
        depth_or_array_layers: depth,
        format,
        bytes,
        label: "3d mip",
    })
}

/// Writes one mip level of a 3D texture while the caller already holds the queue gate.
pub(in crate::assets::texture::upload) fn write_texture3d_volume_mip_with_gate(
    write: &Texture3dVolumeMipWrite<'_>,
    gate: &parking_lot::MutexGuard<'_, ()>,
) -> Result<(), TextureUploadError> {
    profiling::scope!("asset::texture3d_write_volume_mip_locked");
    let Texture3dVolumeMipWrite {
        queue,
        gpu_queue_access_gate,
        queue_access_mode,
        texture,
        mip_level,
        width,
        height,
        depth,
        format,
        bytes,
    } = *write;
    write_texture_region_with_gate(
        TextureRegionWrite {
            queue,
            gpu_queue_access_gate,
            queue_access_mode,
            destination: wgpu::TexelCopyTextureInfo {
                texture,
                mip_level,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            width,
            height,
            depth_or_array_layers: depth,
            format,
            bytes,
            label: "3d mip",
        },
        gate,
    )
}
