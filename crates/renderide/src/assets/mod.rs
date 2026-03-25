//! Asset storage and management.

pub mod manager;
pub mod material_properties;
pub mod mesh;
pub mod registry;
pub mod shader;
pub mod texture;
pub mod texture_unpack;
pub mod ui_material_contract;

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
pub use texture::TextureAsset;
pub use texture_unpack::{
    HostTextureAssetKind, texture2d_asset_id_from_packed, unpack_host_texture_packed,
};
pub use ui_material_contract::{
    NativeUiShaderFamily, UiTextUnlitMaterialUniform, UiTextUnlitPropertyIds, UiUnlitFlags,
    UiUnlitMaterialUniform, UiUnlitPropertyIds, native_ui_family_for_shader,
    native_ui_family_from_shader_path_hint, resolve_native_ui_shader_family,
    ui_text_unlit_material_uniform, ui_unlit_material_uniform,
};
