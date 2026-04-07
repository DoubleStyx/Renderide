//! Host Texture2D ingest: format resolution, mip layout, SHM → [`wgpu::Queue::write_texture`].
//!
//! Does **not** retain CPU pixel buffers after upload (meshes parity). For mip streaming / eviction,
//! see [`crate::resources::GpuTexture2d`] and [`crate::resources::StreamingPolicy`].

mod decode;
mod format;
mod layout;
mod upload;

pub use format::{pick_wgpu_storage_format, supported_host_formats_for_init};
pub use layout::{
    estimate_gpu_texture_bytes, host_format_is_compressed, mip_byte_len, total_mip_chain_byte_len,
    validate_mip_upload_layout,
};
pub use upload::{resolve_texture2d_wgpu_format, write_texture2d_mips};
