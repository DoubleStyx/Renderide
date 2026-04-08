//! Maps host material property ids to future `@group(1)` WGSL bindings (textures, uniform blocks).
//!
//! [`crate::assets::material::MaterialPropertyStore`] and [`crate::assets::material::PropertyIdRegistry`]
//! already capture Unity [`Material`](https://docs.unity3d.com/ScriptReference/Material.html) /
//! [`MaterialPropertyBlock`](https://docs.unity3d.com/ScriptReference/MaterialPropertyBlock.html) data on
//! CPU. GPU upload paths will use per-shader manifests (from `shaders/target/manifest.json` or naga
//! reflection) to pack [`crate::assets::material::MaterialPropertyValue`] into bind group 1.

/// Placeholder for a per-logical-shader layout describing which property ids feed which bindings.
#[derive(Debug, Default)]
pub struct MaterialPropertyGpuLayout {
    /// Reserved until manifests list `property_name` → `binding` pairs.
    pub _pending: (),
}

impl MaterialPropertyGpuLayout {
    /// Builds an empty layout (no GPU bindings yet).
    pub fn empty() -> Self {
        Self::default()
    }
}
