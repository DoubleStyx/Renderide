//! Asset storage and management.

pub mod manager;
#[path = "material/mod.rs"]
pub mod material_support;
pub mod mesh;
pub mod registry;
pub mod shader;
#[path = "shader_support.rs"]
pub mod shader_support;
pub mod texture;
#[path = "texture_support.rs"]
pub mod texture_support;
pub mod util;

pub use material_support as material;
pub use material_support::batch_wire_metrics as material_batch_wire_metrics;
pub use material_support::native_ui_blend;
pub use material_support::properties as material_properties;
pub use material_support::property_host as material_property_host;
pub use material_support::ui_contract as ui_material_contract;
pub use material_support::update_batch as material_update_batch;
pub use material_support::world_unlit_contract as world_unlit_material_contract;
pub use shader_support as shader_meta;
pub use shader_support::host_router as host_shader_router;
pub use shader_support::logical_name as shader_logical_name;
pub use shader_support::program as shader_program;
pub use texture_support::unpack as texture_unpack;

/// Handle used to identify assets across the registry.
pub type AssetId = i32;

/// Trait for assets that can be stored in the registry.
/// Mirrors Unity's asset handle system (Texture2DAsset, MaterialAssetManager, etc.).
pub trait Asset: Send + Sync + 'static {
    /// Returns the unique identifier for this asset.
    fn id(&self) -> AssetId;
}

pub use material_support::native_ui_blend::{
    NativeUiSurfaceBlend, resolve_native_ui_surface_blend_text,
    resolve_native_ui_surface_blend_unlit,
};
pub use material_support::properties::{
    MaterialDictionary, MaterialPropertyLookupIds, MaterialPropertyStore, MaterialPropertyValue,
};
pub use material_support::property_host::{
    apply_froox_material_property_name_to_native_ui_config,
    apply_froox_material_property_name_to_pbr_host_config,
    apply_froox_material_property_name_to_world_unlit_config, intern_host_material_property_id,
};
pub use material_support::ui_contract::{
    DEFAULT_OVERLAY_TINT, NativeUiShaderFamily, UiTextUnlitMaterialUniform, UiTextUnlitPropertyIds,
    UiUnlitFlags, UiUnlitMaterialUniform, UiUnlitPropertyIds, native_ui_family_from_shader_label,
    native_ui_family_from_shader_path_hint, native_ui_family_from_unity_shader_name,
    resolve_native_ui_shader_family, ui_text_unlit_material_uniform, ui_unlit_material_uniform,
};
pub use material_support::world_unlit_contract::{
    WorldUnlitFlags, WorldUnlitMaterialUniform, WorldUnlitPropertyIds, WorldUnlitShaderFamily,
    log_world_unlit_material_inventory_if_enabled, resolve_world_unlit_shader_family,
    world_unlit_family_from_unity_shader_name, world_unlit_material_uniform,
};
pub use mesh::{
    BlendshapeOffset, MeshAsset, attribute_offset_and_size, attribute_offset_size_format,
    compute_vertex_stride,
};
pub use registry::AssetRegistry;
pub use shader::ShaderAsset;
pub use shader_support::host_router::{
    NativeShaderRoute, pbs_metallic_family_from_shader_path_hint,
    pbs_metallic_family_from_unity_shader_name, resolve_native_shader_route,
    resolve_pbs_metallic_shader_family,
};
pub use shader_support::logical_name::{
    CANONICAL_UNITY_UI_TEXT_UNLIT, CANONICAL_UNITY_UI_UNLIT, CANONICAL_UNITY_WORLD_UNLIT,
    parse_shader_lab_quoted_name, parse_wgsl_unity_shader_name_banner,
    resolve_logical_shader_name_from_upload,
    resolve_logical_shader_name_from_upload_with_host_hint,
};
pub use shader_support::program::{
    EssentialShaderProgram, ResolvedRenderideShader, ShaderPipelineFamily,
    resolve_essential_shader_program, resolve_renderide_shader_binding,
    resolve_renderide_shader_rel_path,
};
pub use texture::TextureAsset;
pub use texture_support::unpack::{
    HostTextureAssetKind, texture2d_asset_id_from_packed, unpack_host_texture_packed,
};
