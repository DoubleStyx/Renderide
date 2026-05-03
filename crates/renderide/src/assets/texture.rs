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

pub use format::{pick_wgpu_storage_format, supported_host_formats_for_init};
pub use layout::{
    estimate_gpu_cubemap_bytes, estimate_gpu_texture_bytes, estimate_gpu_texture3d_bytes,
    host_format_is_compressed, mip_byte_len, mip_dimensions_at_level_3d, mip_tight_bytes_per_texel,
    total_mip_chain_byte_len, total_mip_chain_volume_byte_len, validate_mip_upload_layout,
};
pub use unpack::{
    HostTextureAssetKind, texture2d_asset_id_from_packed, unpack_host_texture_packed,
};
pub(crate) use upload::upload_uses_storage_v_inversion;
pub use upload::{
    CubemapFaceMipUploadStep, CubemapMipChainUploader, MipChainAdvance, Texture2dUploadContext,
    Texture3dMipAdvance, Texture3dMipChainUploader, Texture3dMipUploadStep, Texture3dUploadContext,
    TextureDataStart, TextureMipChainUploader, TextureMipUploadStep, TextureUploadError,
    resolve_cubemap_wgpu_format, resolve_texture2d_wgpu_format, resolve_texture3d_wgpu_format,
    texture_upload_start, write_texture2d_mips, write_texture3d_mips,
};
