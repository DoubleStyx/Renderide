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
//! * `test_support` (test-only) -- builders shared by submodule unit tests.

pub(crate) mod budget;
pub(crate) mod pools;
pub(crate) mod resource_pool;
pub(crate) mod sampler_state;
pub(crate) mod texture_allocation;

#[cfg(test)]
pub(crate) mod test_support;

pub(crate) use budget::{NoopStreamingPolicy, StreamingPolicy, VramAccounting, VramResourceKind};
pub(crate) use pools::cubemap::{CubemapPool, GpuCubemap};
pub(crate) use pools::mesh::MeshPool;
pub(crate) use pools::render_texture::{GpuRenderTexture, RenderTexturePool};
pub(crate) use pools::texture2d::{GpuTexture2d, TexturePool};
pub(crate) use pools::texture3d::{GpuTexture3d, Texture3dPool};
pub(crate) use pools::video_texture::{GpuVideoTexture, VideoTexturePool};
pub(crate) use sampler_state::SamplerState;

/// Common surface for resident GPU resources (extend for textures, buffers, etc.).
pub(crate) trait GpuResource {
    /// Approximate GPU memory for accounting.
    fn resident_bytes(&self) -> u64;
    /// Host asset id.
    fn asset_id(&self) -> i32;
}

/// Implements [`GpuResource`] for a type with `resident_bytes: u64` and
/// `asset_id: i32` inherent fields.
macro_rules! impl_gpu_resource {
    ($ty:ty) => {
        impl $crate::gpu_pools::GpuResource for $ty {
            #[inline]
            fn resident_bytes(&self) -> u64 {
                self.resident_bytes
            }

            #[inline]
            fn asset_id(&self) -> i32 {
                self.asset_id
            }
        }
    };
}

pub(crate) use impl_gpu_resource;
