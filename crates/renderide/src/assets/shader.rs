//! Shader asset type. Stub for host-uploaded WGSL shaders.
//!
//! Extension point for shader_upload command handling.

use super::Asset;
use super::AssetId;

/// Stored shader data for pipeline creation.
/// Stub: WGSL source populated when shader_upload is implemented.
pub struct ShaderAsset {
    /// Unique identifier for this shader.
    pub id: AssetId,
    /// Optional WGSL source. Populated when shader_upload provides it.
    pub wgsl_source: Option<String>,
}

impl Asset for ShaderAsset {
    fn id(&self) -> AssetId {
        self.id
    }
}
