//! GPU resource pools and VRAM hooks (meshes, Texture2D, Texture3D, cubemaps, video textures).

mod budget;
mod cubemap_pool;
mod mesh_pool;
mod render_texture_pool;
mod resource_pool;
mod texture3d_pool;
mod texture_allocation;
mod texture_pool;
mod video_texture_pool;

pub use budget::{
    MeshResidencyMeta, NoopStreamingPolicy, ResidencyTier, StreamingPolicy, TextureResidencyMeta,
    VramAccounting, VramResourceKind,
};
pub use cubemap_pool::{CubemapPool, CubemapSamplerState, GpuCubemap};
pub use mesh_pool::MeshPool;
pub use render_texture_pool::{GpuRenderTexture, RenderTexturePool};
pub use texture_pool::{GpuTexture2d, Texture2dSamplerState, TexturePool};
pub use texture3d_pool::{GpuTexture3d, Texture3dPool, Texture3dSamplerState};
pub use video_texture_pool::{GpuVideoTexture, VideoTexturePool};

/// Common surface for resident GPU resources (extend for textures, buffers, etc.).
pub trait GpuResource {
    /// Approximate GPU memory for accounting.
    fn resident_bytes(&self) -> u64;
    /// Host asset id.
    fn asset_id(&self) -> i32;
}
