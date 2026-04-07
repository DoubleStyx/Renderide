//! Re-exports shader modules: host routing, logical names, programs, and Unity upload helpers.

#[path = "shader/host_router.rs"]
pub mod host_router;
#[path = "shader/logical_name.rs"]
pub mod logical_name;
#[path = "shader/program.rs"]
pub mod program;
#[path = "shader/unity_asset.rs"]
pub(crate) mod unity_asset;

pub use super::material_support::ui_contract::{
    NativeUiShaderFamily, native_ui_family_from_unity_shader_name,
};
pub use super::material_support::world_unlit_contract::world_unlit_family_from_unity_shader_name;
pub use super::{AssetRegistry, ShaderAsset};
pub use logical_name as shader_logical_name;
pub use program::{EssentialShaderProgram, ShaderPipelineFamily};
pub(crate) use unity_asset as shader_unity_asset;
