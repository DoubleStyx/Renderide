//! Pure asset ingestion helpers: mesh layouts, texture formats, shader routing, and video decode.
//!
//! # Module map
//!
//! - **`mesh`** -- Host [`mesh::MeshBufferLayout`] contract, [`mesh::GpuMesh`] construction, layout
//!   fingerprints, and upload validation. [`crate::gpu_pools::GpuResource`] is implemented for resident meshes.
//! - **`shader`** -- Resolving [`crate::shared::ShaderUpload`] AssetBundle paths to pipeline kinds for
//!   [`crate::materials::MaterialRegistry`].
//! - **`texture`** -- Host Texture2D format/layout, decode/swizzle, mip packing, and
//!   [`wgpu::Queue::write_texture`] uploads.
//! - **`util`** -- Small string helpers shared with [`crate::materials`] (e.g. Unity shader key normalization).

pub mod mesh;
pub mod shader;
pub mod texture;
pub mod util;
pub mod video;

pub use shader::{ResolvedShaderUpload, resolve_shader_upload};
