//! Texture2D full-mip upload wrappers.

use super::TextureUploadError;
use super::region::{TextureRegionWrite, write_texture_region, write_texture_region_with_gate};
use crate::gpu::GpuQueueAccessMode;

/// Descriptor for [`write_one_mip`]: one mip of a 2D texture via [`wgpu::Queue::write_texture`].
pub(in crate::assets::texture::upload) struct Texture2dMipWrite<'a> {
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
    /// Texel format (must match texture creation).
    pub format: wgpu::TextureFormat,
    /// Tightly packed mip bytes.
    pub bytes: &'a [u8],
}

/// Writes one full 2D mip level.
pub(in crate::assets::texture::upload) fn write_one_mip(
    write: &Texture2dMipWrite<'_>,
) -> Result<(), TextureUploadError> {
    profiling::scope!("asset::texture_write_mip");
    let Texture2dMipWrite {
        queue,
        gpu_queue_access_gate,
        queue_access_mode,
        texture,
        mip_level,
        width,
        height,
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
        depth_or_array_layers: 1,
        format,
        bytes,
        label: "mip",
    })
}

/// Writes one full 2D mip level while the caller already holds the queue gate.
pub(in crate::assets::texture::upload) fn write_one_mip_with_gate(
    write: &Texture2dMipWrite<'_>,
    gate: &parking_lot::MutexGuard<'_, ()>,
) -> Result<(), TextureUploadError> {
    profiling::scope!("asset::texture_write_mip_locked");
    let Texture2dMipWrite {
        queue,
        gpu_queue_access_gate,
        queue_access_mode,
        texture,
        mip_level,
        width,
        height,
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
            depth_or_array_layers: 1,
            format,
            bytes,
            label: "mip",
        },
        gate,
    )
}
