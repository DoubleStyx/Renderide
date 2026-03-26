//! Shader asset type for host-uploaded WGSL (or future intermediate representations).
//!
//! Filled by [`super::AssetRegistry::handle_shader_upload`].

use super::Asset;
use super::AssetId;

/// Stored shader data for pipeline creation.
pub struct ShaderAsset {
    /// Unique identifier for this shader.
    pub id: AssetId,
    /// Optional path or inline source string from the host `ShaderUpload.file` field.
    pub wgsl_source: Option<String>,
    /// Unity ShaderLab logical name (`Shader "UI/Unlit"`) from parsed ShaderLab/WGSL text, file contents,
    /// or an optional host hint when your IPC layer supplies one (see [`crate::shared::shader_upload_extras`]).
    pub unity_shader_name: Option<String>,
}

impl Asset for ShaderAsset {
    fn id(&self) -> AssetId {
        self.id
    }
}
