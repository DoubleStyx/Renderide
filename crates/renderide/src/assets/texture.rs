//! Host texture ingest: format resolution, mip layout, SHM -> [`wgpu::Queue::write_texture`].
//!
//! Covers **Texture2D**, **Texture3D**, and **cubemap** uploads. Does **not** retain CPU pixel buffers
//! after upload (meshes parity). For mip streaming / eviction, see [`crate::gpu_pools::GpuTexture2d`]
//! and [`crate::gpu_pools::StreamingPolicy`].

mod decode;
mod format;
mod layout;
#[cfg(test)]
mod layout_tests;
mod unpack;
mod upload;

pub use format::supported_host_formats_for_init;
pub use layout::{
    estimate_gpu_cubemap_bytes, estimate_gpu_texture_bytes, estimate_gpu_texture3d_bytes,
    host_texture_mip_count, legal_texture2d_mip_level_count, legal_texture3d_mip_level_count,
};
pub use unpack::{
    HostTextureAssetKind, texture2d_asset_id_from_packed, unpack_host_texture_packed,
};
pub(crate) use upload::upload_uses_storage_v_inversion;
pub use upload::{
    CubemapFaceMipUploadStep, CubemapMipChainUploader, MipChainAdvance, Texture2dUploadInputs,
    Texture2dUploadPayload, Texture2dUploadQueueInputs, Texture2dUploadTarget, Texture3dMipAdvance,
    Texture3dMipChainUploader, Texture3dMipUploadStep, TextureDataStart, TextureMipChainUploader,
    TextureMipUploadStep, TextureUploadError, resolve_cubemap_wgpu_format,
    resolve_texture2d_wgpu_format, resolve_texture3d_wgpu_format, texture_upload_start,
};
