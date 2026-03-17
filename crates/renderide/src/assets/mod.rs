//! Asset storage and management.

pub mod manager;
pub mod material_properties;
pub mod mesh;
pub mod registry;
pub mod shader;
pub mod texture;

/// Handle used to identify assets across the registry.
pub type AssetId = i32;

/// Trait for assets that can be stored in the registry.
/// Mirrors Unity's asset handle system (Texture2DAsset, MaterialAssetManager, etc.).
pub trait Asset: Send + Sync + 'static {
    /// Returns the unique identifier for this asset.
    fn id(&self) -> AssetId;
}

pub use material_properties::{MaterialPropertyStore, MaterialPropertyValue};
pub use mesh::{
    BlendshapeOffset, MeshAsset, attribute_offset_and_size, attribute_offset_size_format,
    compute_vertex_stride,
};
pub use registry::AssetRegistry;
pub use shader::ShaderAsset;
