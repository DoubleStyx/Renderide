//! Asset storage and management.

pub mod mesh;
pub mod registry;

pub use mesh::{attribute_offset_and_size, attribute_offset_size_format, compute_vertex_stride, MeshAsset};
pub use registry::AssetRegistry;
