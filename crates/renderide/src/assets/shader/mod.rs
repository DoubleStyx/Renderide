//! Host [`ShaderUpload`](crate::shared::ShaderUpload) handling: logical name extraction and material routing.

pub mod logical_name;
pub mod route;
pub mod unity_asset;

pub use route::{
    classify_shader, resolve_shader_upload, CoarseShaderKind, ResolvedShaderUpload, UiFamily,
    WorldUnlitFamily,
};
