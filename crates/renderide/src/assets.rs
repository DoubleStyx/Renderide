//! Asset ingestion, shader routing helpers, and GPU upload orchestration.
//!
//! # Module map
//!
//! - **`asset_transfer_queue`** -- [`AssetTransferQueue`]: IPC-driven mesh/texture/render-texture
//!   queues, per-poll upload budgets, CPU-side format/property tables, and
//!   [`crate::gpu_pools::MeshPool`] / [`crate::gpu_pools::TexturePool`] / [`crate::gpu_pools::RenderTexturePool`].
//!   Owned by [`crate::backend::RenderBackend`] after GPU attach.
//! - **`mesh`** -- Host [`mesh::MeshBufferLayout`] contract, [`mesh::GpuMesh`] construction, layout
//!   fingerprints, and upload validation. [`crate::gpu_pools::GpuResource`] is implemented for resident meshes.
//! - **`shader`** -- Resolving [`crate::shared::ShaderUpload`] AssetBundle paths to pipeline kinds for
//!   [`crate::materials::MaterialRegistry`].
//! - **`texture`** -- Host Texture2D format/layout, decode/swizzle, mip packing, and
//!   [`wgpu::Queue::write_texture`] uploads.
//! - **`util`** -- Small string helpers shared with [`crate::materials`] (e.g. Unity shader key normalization).

pub mod asset_transfer_queue;
pub mod mesh;
pub mod shader;
pub mod texture;
pub mod util;
pub mod video;

pub use asset_transfer_queue::AssetTransferQueue;
pub use shader::{ResolvedShaderUpload, resolve_shader_upload};

/// Alias for [`AssetTransferQueue`] (same type: upload queues, pools, and per-poll budgets).
pub type AssetSubsystem = AssetTransferQueue;
