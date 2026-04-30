//! Host [`ShaderUpload`](crate::shared::ShaderUpload) handling: AssetBundle shader-name extraction and material routing.

pub mod route;
pub mod unity_asset;

pub use route::{resolve_shader_upload, ResolvedShaderUpload};
