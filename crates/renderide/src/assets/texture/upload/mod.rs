//! Applies [`SetTexture2DData`](crate::shared::SetTexture2DData) into an existing [`wgpu::Texture`] using [`wgpu::Queue::write_texture`]
//! ([`wgpu::TexelCopyTextureInfo`] / [`wgpu::TexelCopyBufferLayout`]).
//!
//! When [`crate::shared::TextureUploadHint::has_region`] is set with a non-empty rectangle, mip0 may be
//! uploaded as a sub-rect only (uncompressed RGBA8-family GPU storage, single mip, no `flip_y`).
//! Other cases fall back to the full mip chain path.
//!
//! The [`wgpu::TextureFormat`] must match the texture’s creation format (see [`format_resolve::resolve_texture2d_wgpu_format`]).

mod cubemap_write;
mod error;
mod format_resolve;
mod mip_write_common;
mod subregion;
mod texture3d_write;
mod write_mip_chain;

pub use cubemap_write::{CubemapFaceMipUploadStep, CubemapMipChainUploader};
pub use error::TextureUploadError;
pub use format_resolve::{
    resolve_cubemap_wgpu_format, resolve_texture2d_wgpu_format, resolve_texture3d_wgpu_format,
};
pub use texture3d_write::{
    write_texture3d_mips, Texture3dMipAdvance, Texture3dMipChainUploader, Texture3dMipUploadStep,
    Texture3dUploadContext,
};
pub use write_mip_chain::{
    texture_upload_start, write_texture2d_mips, MipChainAdvance, Texture2dUploadContext,
    TextureDataStart, TextureMipChainUploader, TextureMipUploadStep,
};
