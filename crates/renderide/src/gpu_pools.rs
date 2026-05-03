//! GPU resource pools and VRAM hooks (meshes, Texture2D, Texture3D, cubemaps, render textures, video textures).
//!
//! ## Module layout
//!
//! * [`budget`] -- VRAM accounting, residency tiers, streaming policy trait, residency-meta hints.
//! * [`resource_pool`] -- generic `GpuResourcePool<T, A>` + `PoolResourceAccess` trait + the two
//!   facade macros (streaming vs untracked).
//! * [`sampler_state`] -- unified [`SamplerState`] consumed by every texture-bearing pool and the
//!   material bind layer.
//! * [`texture_allocation`] -- `wgpu::Texture` + `wgpu::TextureView` factory shared by the three
//!   sampled-texture pools.
//! * [`pools`] -- concrete pool newtypes, one submodule per asset kind.

pub(crate) mod budget;
pub(crate) mod pools;
pub(crate) mod resource_pool;
pub(crate) mod sampler_state;
pub(crate) mod texture_allocation;

pub use budget::{
    MeshResidencyMeta, NoopStreamingPolicy, ResidencyTier, StreamingPolicy, TextureResidencyMeta,
    VramAccounting, VramResourceKind,
};
pub use pools::cubemap::{CubemapPool, GpuCubemap};
pub use pools::mesh::MeshPool;
pub use pools::render_texture::{GpuRenderTexture, RenderTexturePool};
pub use pools::texture2d::{GpuTexture2d, TexturePool};
pub use pools::texture3d::{GpuTexture3d, Texture3dPool};
pub use pools::video_texture::{GpuVideoTexture, VideoTexturePool};
pub use sampler_state::SamplerState;

/// Common surface for resident GPU resources (extend for textures, buffers, etc.).
pub trait GpuResource {
    /// Approximate GPU memory for accounting.
    fn resident_bytes(&self) -> u64;
    /// Host asset id.
    fn asset_id(&self) -> i32;
}
