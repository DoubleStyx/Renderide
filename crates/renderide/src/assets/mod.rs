//! Asset storage and management.

pub mod manager;
pub mod material_batch_wire_metrics;
pub mod material_properties;
pub mod material_property_host;
pub mod material_update_batch;
pub mod mesh;
pub mod native_ui_blend;
pub mod registry;
pub mod shader;
pub mod shader_logical_name;
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

pub use material_properties::{
    MaterialPropertyLookupIds, MaterialPropertyStore, MaterialPropertyValue,
};
pub use material_property_host::{
    apply_froox_material_property_name_to_native_ui_config,
    apply_froox_material_property_name_to_pbr_host_config, intern_host_material_property_id,
};
pub use mesh::{
    BlendshapeOffset, MeshAsset, attribute_offset_and_size, attribute_offset_size_format,
    compute_vertex_stride,
};
pub use native_ui_blend::{
    NativeUiSurfaceBlend, resolve_native_ui_surface_blend_text,
    resolve_native_ui_surface_blend_unlit,
};
pub use registry::AssetRegistry;
pub use shader::ShaderAsset;
pub use shader_logical_name::{
    CANONICAL_UNITY_UI_TEXT_UNLIT, CANONICAL_UNITY_UI_UNLIT, parse_shader_lab_quoted_name,
    parse_wgsl_unity_shader_name_banner, resolve_logical_shader_name_from_upload,
    resolve_logical_shader_name_from_upload_with_host_hint,
};
pub use texture::TextureAsset;
pub use texture_unpack::{
    HostTextureAssetKind, texture2d_asset_id_from_packed, unpack_host_texture_packed,
};
pub use ui_material_contract::{
    NativeUiShaderFamily, UiTextUnlitMaterialUniform, UiTextUnlitPropertyIds, UiUnlitFlags,
    UiUnlitMaterialUniform, UiUnlitPropertyIds, native_ui_family_for_shader,
    native_ui_family_from_shader_label, native_ui_family_from_shader_path_hint,
    native_ui_family_from_unity_shader_name, resolve_native_ui_shader_family,
    ui_text_unlit_material_uniform, ui_unlit_material_uniform,
};
