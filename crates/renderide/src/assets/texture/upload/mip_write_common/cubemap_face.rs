//! Cubemap face-mip upload wrappers.

use super::TextureUploadError;
use super::region::{TextureRegionWrite, write_texture_region, write_texture_region_with_gate};
use crate::gpu::GpuQueueAccessMode;

/// Descriptor for [`write_cubemap_face_mip`]: one cubemap face x one mip (2D array layer).
pub(in crate::assets::texture::upload) struct CubemapFaceMipWrite<'a> {
    /// Queue used for the texel copy.
    pub queue: &'a wgpu::Queue,
    /// Shared GPU queue access gate for [`wgpu::Queue::write_texture`]; see
    /// [`crate::gpu::GpuQueueAccessGate`].
    pub gpu_queue_access_gate: &'a crate::gpu::GpuQueueAccessGate,
    /// Queue-gate acquisition policy for this write.
    pub queue_access_mode: GpuQueueAccessMode,
    /// Destination cubemap texture (`D2` array with six layers).
    pub texture: &'a wgpu::Texture,
    /// Mip level index.
    pub mip_level: u32,
    /// Array layer index `0..6` for the cube face.
    pub face_layer: u32,
    /// Face width in texels.
    pub width: u32,
    /// Face height in texels.
    pub height: u32,
    /// Texel format (must match texture creation).
    pub format: wgpu::TextureFormat,
    /// Tightly packed mip bytes for this face.
    pub bytes: &'a [u8],
}

/// Writes one face x one mip of a cubemap (`D2` texture with six array layers).
pub(in crate::assets::texture::upload) fn write_cubemap_face_mip(
    write: &CubemapFaceMipWrite<'_>,
) -> Result<(), TextureUploadError> {
    profiling::scope!("asset::cubemap_write_face_mip");
    let CubemapFaceMipWrite {
        queue,
        gpu_queue_access_gate,
        queue_access_mode,
        texture,
        mip_level,
        face_layer,
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
            origin: wgpu::Origin3d {
                x: 0,
                y: 0,
                z: face_layer,
            },
            aspect: wgpu::TextureAspect::All,
        },
        width,
        height,
        depth_or_array_layers: 1,
        format,
        bytes,
        label: "cubemap mip",
    })
}

/// Writes one face x one mip of a cubemap while the caller already holds the queue gate.
pub(in crate::assets::texture::upload) fn write_cubemap_face_mip_with_gate(
    write: &CubemapFaceMipWrite<'_>,
    gate: &parking_lot::MutexGuard<'_, ()>,
) -> Result<(), TextureUploadError> {
    profiling::scope!("asset::cubemap_write_face_mip_locked");
    let CubemapFaceMipWrite {
        queue,
        gpu_queue_access_gate,
        queue_access_mode,
        texture,
        mip_level,
        face_layer,
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
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: face_layer,
                },
                aspect: wgpu::TextureAspect::All,
            },
            width,
            height,
            depth_or_array_layers: 1,
            format,
            bytes,
            label: "cubemap mip",
        },
        gate,
    )
}
